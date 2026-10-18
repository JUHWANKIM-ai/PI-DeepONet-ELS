# -*- coding: utf-8 -*-
"""torch MC 엔진 (CUDA 기본 / CPU 폴백) — 페이오프 규칙의 정본 구현.

노트북 4의 V3 이식: 시간축 블록 + carry 로 경로를 만들고, 선형성
cumsum(drift + s*Z) = cumdrift + s*cumsum(Z) 로 drift pass 를 없앴으며,
worst-of 비교는 log-space 에서 한다 (min exp(x) >= K  <=>  min x >= log K).
관측일 판정은 argmax(first event) 로 완전 벡터화. 상품당 약 27 ms (RTX 3060 Ti).

규칙 우선순위: 조기상환(회차별 지급률) > 리자드(FROM_ISU no-touch, 축소 지급률) > 만기(낙인 판정).
B >= 1.0 은 노낙인 = 만기 미상환이면 항상 worst 수취."""
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from . import features as F
from .mc import NPATH, MC_COLS, prepare_one

DEFAULT_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def payoff_from_path_summary_t(
    wobs: torch.Tensor,
    wrun: torch.Tensor | None,
    wmin: torch.Tensor,
    wT: torch.Tensor,
    logK: torch.Tensor,
    rate: torch.Tensor,
    lz_on: torch.Tensor,
    loglzb: torch.Tensor,
    lzp: torch.Tensor,
    always_ki: bool,
    logB: float,
    DF: torch.Tensor,
    obs_day: torch.Tensor,
) -> torch.Tensor:
    """경로요약 -> 할인 페이오프. mc_daily_t 가 쓰는 페이오프 블록을 그대로 함수로 분리한 것.

    PI 물리항(module.pi_physics)이 '실측 MC 규약'과 일치하는지 테스트에서 직접 대조하려고 노출한다.
    정규 상환이 리자드보다 우선. 만기 미상환 시 노낙인(B>=1)은 항상 worst, 낙인은 배리어 터치 여부.

    입력 (전부 log 공간. m=경로 수, nobs=관측 회차)
      wobs      : Tensor(m, nobs) — 각 관측일의 log worst-of
      wrun      : Tensor(m, nobs) | None — 발행 이후 러닝 최저. 리자드가 없으면 None
      wmin      : Tensor(m,) — 전 구간 최저 (낙인 판정용)
      wT        : Tensor(m,) — 만기 log worst-of
      logK      : Tensor(nobs,) — 회차별 log 행사가
      rate      : Tensor(nobs,) — 회차별 정규 지급률
      lz_on     : Tensor(nobs,) bool — 그 회차에 리자드 조항이 있는가
      loglzb    : Tensor(nobs,) — log 리자드 배리어 (조항 없는 회차는 log 1 = 0)
      lzp       : Tensor(nobs,) — 리자드 지급률
      always_ki : bool — B >= 1.0 (노낙인). True 면 만기에 항상 worst 를 받는다
      logB      : float — log KI 배리어
      DF        : Tensor(N,) — 일별 할인계수
      obs_day   : Tensor(nobs,) long — 회차별 경과일 (1-기반)
    출력
      Tensor(m,) — 경로별 할인 페이오프
    """
    f32 = wobs.dtype
    hit = wobs >= logK
    has_lz = bool(lz_on.any())
    ev = (hit | ((wrun >= loglzb) & lz_on)) if has_lz else hit
    any_ev = ev.any(dim=1)
    first = ev.to(f32).argmax(dim=1)
    is_reg = hit.gather(1, first[:, None]).squeeze(1)
    pr = torch.where(is_reg, rate[first], lzp[first])  # 정규 상환이 리자드보다 우선
    early = (1.0 + pr) * DF[obs_day[first] - 1]
    eT = torch.exp(wT)
    sv = eT if always_ki else torch.where(wmin < logB, eT, torch.ones_like(eT))
    return torch.where(any_ev, early, sv * DF[-1])


def _zero(
    beta: np.ndarray | Callable[[npt.ArrayLike], np.ndarray], times: np.ndarray
) -> np.ndarray:
    """무위험금리를 두 표현 중 무엇이 와도 같은 방식으로 읽는다.

    입력
      beta  : ndarray(3,) | callable — NS 계수 벡터, 또는 '만기배열 → 금리' 콜러블
              (`curve.curve_dz` 의 부트스트랩 커브)
      times : ndarray — 만기 [년]
    출력
      ndarray — 같은 길이의 연속복리 제로금리 (소수)
    """
    return np.asarray(beta(times), float) if callable(beta) else F.zero_curve(beta, times)


def _setup(
    sigs: Sequence[float],
    corr: np.ndarray,
    beta: np.ndarray | Callable[[npt.ArrayLike], np.ndarray],
    B: float,
    strikes: Sequence[float],
    ten: float,
    c: float,
    pmts: Sequence[float] | None,
    lz_barr: Sequence[float] | None,
    lz_pmt: Sequence[float] | None,
    qcont: npt.ArrayLike | None = None,
    divlog: np.ndarray | None = None,
    disc_beta: np.ndarray | Callable[[npt.ArrayLike], np.ndarray] | None = None,
) -> dict[str, Any]:
    """경로 생성에 필요한 상수를 미리 전부 numpy 로 만들어 둔다 (장치 전송 전 단계).

    드리프트·할인·배당·상관·리자드 마스크를 여기서 한 번에 조립하므로, 배당 소스나 커브를
    바꾸는 변형은 전부 이 함수의 인자로 들어온다.

    입력
      sigs       : list[float] 길이 3 — 연율 변동성
      corr       : ndarray(3,3) — 상관행렬
      beta       : ndarray(3,) | callable — **드리프트**에 쓸 커브
      B          : float — KI 배리어
      strikes    : list[float] — 회차별 행사가. 길이가 회차 수가 된다
      ten        : float — 만기 [년]
      c          : float — 연쿠폰. pmts 가 None 일 때만 쓰인다
      pmts       : list[float] | None — 회차별 지급률. None 이면 c·경과연수로 만든다
      lz_barr    : list[float] | None — 회차별 리자드 배리어. NaN 이면 그 회차엔 조항이 없다
      lz_pmt     : list[float] | None — 회차별 리자드 지급률. NaN 은 0 으로 바꾼다
      qcont      : array-like(3,) | None — 연속 배당수익률. 매일 −q·dt 를 드리프트에서 뺀다
      divlog     : ndarray(N,3) | None — 이산배당 로그하락. 드리프트에 더한다
      disc_beta  : ndarray | callable | None — **할인만** 다른 커브 (요인분해용).
                   None 이면 beta 로 할인한다
    출력
      dict — N · nobs · obs_day · DF · L(촐레스키) · sig · drift(N,3) · sq · logB ·
             logK · rate · lzp · lz_on · loglzb. 전부 float32 numpy 다
    """
    nobs = len(strikes)
    N = int(round(ten * 365))
    dt = 1 / 365
    obs_day = np.clip(np.round(np.arange(1, nobs + 1) * (ten / nobs) * 365).astype(int), 1, N)
    obs_t = obs_day / 365.0
    times = np.arange(1, N + 1) * dt
    DF = np.exp(-_zero(beta, times) * times)
    fdt = (-(np.diff(np.concatenate([[0.0], np.log(DF)])))).astype(np.float32)
    sig = np.asarray(sigs, np.float32)
    drift = (fdt[:, None] - (0.5 * sig**2 * np.float32(dt))[None, :]).astype(np.float32)
    if qcont is not None:  # 연속 배당수익률: 매일 −q·dt
        drift = drift - (np.asarray(qcont, np.float32) / np.float32(365.0))[None, :]
    if divlog is not None:  # 이산배당: 배당락일에 log(1−δ) (N×3)
        drift = drift + np.asarray(divlog, np.float32)
    if disc_beta is not None:  # 할인만 다른 커브 (드리프트는 beta) — 요인분해용
        DF = np.exp(-_zero(disc_beta, times) * times)
    rate = np.asarray(c * obs_t if pmts is None else pmts, np.float32)[:nobs]
    lzb = (
        np.full(nobs, np.nan, np.float32)
        if lz_barr is None
        else np.asarray(lz_barr, np.float32)[:nobs]
    )
    lzp = (
        np.zeros(nobs, np.float32)
        if lz_pmt is None
        else np.nan_to_num(np.asarray(lz_pmt, np.float32))[:nobs]
    )
    on = np.isfinite(lzb)
    return dict(
        N=N,
        nobs=nobs,
        obs_day=obs_day,
        DF=DF.astype(np.float32),
        L=F.chol_psd(np.asarray(corr)).astype(np.float32),
        sig=sig,
        drift=drift,
        sq=np.float32(np.sqrt(dt)),
        logB=float(np.log(B)),
        logK=np.log(np.asarray(strikes, np.float32)),
        rate=rate,
        lzp=lzp,
        lz_on=on,
        loglzb=np.log(np.where(on, lzb, 1.0)).astype(np.float32),
    )


def mc_daily_t(
    sigs: Sequence[float],
    corr: np.ndarray,
    beta: np.ndarray | Callable[[npt.ArrayLike], np.ndarray],
    B: float,
    strikes: Sequence[float],
    ten: float,
    n: int = NPATH,
    seed: int = 0,
    c: float = 0.0,
    pmts: Sequence[float] | None = None,
    lz_barr: Sequence[float] | None = None,
    lz_pmt: Sequence[float] | None = None,
    dev: str | None = None,
    path_chunk: int = 20_000,
    tblock: int = 512,
    qcont: npt.ArrayLike | None = None,
    divlog: np.ndarray | None = None,
    disc_beta: np.ndarray | Callable[[npt.ArrayLike], np.ndarray] | None = None,
) -> float:
    """일별 MC (torch). pmts=None 이면 (1 + c*경과연수), 주어지면 (1 + pmts[j]).
    lz_barr[j] 가 유한하면 정규 조건 실패 경로에 대해 발행 이후 최저 worst >= 배리어일 때 리자드 상환.

    입력
      sigs, corr, beta, B, strikes, ten, c, pmts, lz_barr, lz_pmt, qcont, divlog, disc_beta
                 : `_setup` 과 같다. 기본값은 전부 None 이라 종전 동작과 완전히 같다
      n          : int — 경로 수. 기본 NPATH(100,000)
      seed       : int — 공통난수 시드. 같은 시드·같은 장치면 비트 단위로 재현된다
      dev        : str | None — torch 장치. None 이면 CUDA 가용 시 CUDA
      path_chunk : int — 한 번에 생성할 경로 수 (메모리 상한)
      tblock     : int — 시간축 블록 길이. carry 로 블록 간 누적을 이어 간다
    출력
      float — 액면 1 기준 현재가치. 누적은 float64 로 받아 경로 수가 많아도 정밀도가 유지된다
    """
    P = _setup(
        sigs,
        corr,
        beta,
        B,
        strikes,
        ten,
        c,
        pmts,
        lz_barr,
        lz_pmt,
        qcont=qcont,
        divlog=divlog,
        disc_beta=disc_beta,
    )
    d = torch.device(dev or DEFAULT_DEV)
    f32 = torch.float32
    N, nobs = P["N"], P["nobs"]
    L = torch.as_tensor(P["L"] * (P["sig"] * P["sq"])[:, None], device=d)  # s*sqrt(dt) 를 L 에 흡수
    cdrift = torch.as_tensor(P["drift"], device=d).cumsum(0).T.contiguous()  # (3,N) 누적 drift
    DF = torch.as_tensor(P["DF"], device=d)
    logK = torch.as_tensor(P["logK"], device=d)
    rate = torch.as_tensor(P["rate"], device=d)
    lzp = torch.as_tensor(P["lzp"], device=d)
    loglzb = torch.as_tensor(P["loglzb"], device=d)
    lz_on = torch.as_tensor(P["lz_on"], device=d)
    obs_day = torch.as_tensor(P["obs_day"], device=d, dtype=torch.long)
    has_lz = bool(P["lz_on"].any())
    always_ki = bool(B >= 1.0)
    logB = P["logB"]
    g = torch.Generator(device=d).manual_seed(int(seed))
    tot = torch.zeros((), device=d, dtype=torch.float64)
    done = 0
    while done < n:
        m = min(path_chunk, n - done)
        carry = torch.zeros(3, m, 1, device=d)
        wmin = torch.full((m,), float("inf"), device=d)
        wobs = torch.empty(m, nobs, device=d)
        wrun = torch.empty(m, nobs, device=d) if has_lz else None
        for t0 in range(0, N, tblock):
            T = min(tblock, N - t0)
            X = torch.randn(3, m, T, device=d, dtype=f32, generator=g)
            X[2] = L[2, 0] * X[0] + L[2, 1] * X[1] + L[2, 2] * X[2]  # 하삼각 상관(역순 in-place)
            X[1] = L[1, 0] * X[0] + L[1, 1] * X[1]
            X[0] = L[0, 0] * X[0]
            X = X.cumsum(2)
            X += carry
            carry = X[:, :, -1:].clone()  # drift 제외 누적
            X += cdrift[:, None, t0 : t0 + T]
            w = torch.minimum(torch.minimum(X[0], X[1]), X[2])  # (m,T) log worst-of
            if has_lz:
                cmin = torch.minimum(torch.cummin(w, dim=1).values, wmin[:, None])
            wmin = torch.minimum(wmin, w.amin(dim=1))
            sel = ((obs_day - 1 >= t0) & (obs_day - 1 < t0 + T)).nonzero(as_tuple=True)[0]
            if sel.numel():
                col = obs_day[sel] - 1 - t0
                wobs[:, sel] = w[:, col]
                if has_lz:
                    wrun[:, sel] = cmin[:, col]
        wT = torch.minimum(
            torch.minimum(carry[0, :, 0] + cdrift[0, -1], carry[1, :, 0] + cdrift[1, -1]),
            carry[2, :, 0] + cdrift[2, -1],
        )
        payoff = payoff_from_path_summary_t(
            wobs, wrun, wmin, wT, logK, rate, lz_on, loglzb, lzp, always_ki, logB, DF, obs_day
        )
        tot += payoff.sum(dtype=torch.float64)
        done += m
    return float(tot.item() / n)


def price_one_t(
    mk: dict[str, Any],
    kmap: dict[str, list[float]] | None,
    item: str,
    iord: int,
    B: float,
    c: float,
    ten: float,
    sig_eff: float,
    n: int = NPATH,
    seed: int = 0,
    strikes: Sequence[float] | None = None,
    pmts: Sequence[float] | None = None,
    lz_barr: Sequence[float] | None = None,
    lz_pmt: Sequence[float] | None = None,
    dev: str | None = None,
) -> tuple[float, ...]:
    """상품 하나를 torch 엔진(정본)으로 가격한다. 반환 형태는 `mc.price_one` 과 같다.

    입력
      mk, kmap, item, iord, ten, sig_eff, strikes : `mc.prepare_one` 과 같다
      B       : float — KI 배리어
      c       : float — 연쿠폰
      n       : int — 경로 수
      seed    : int — 공통난수 시드
      pmts    : list | None — 회차별 지급률
      lz_barr : list | None — 리자드 배리어
      lz_pmt  : list | None — 리자드 지급률
      dev     : str | None — torch 장치
    출력
      tuple 길이 9 — MC_COLS 순서: (mc, vol1, vol2, vol3, rho12, rho13, rho23, r_krw, k).
        입력을 못 만들면 전부 np.nan
    """
    P = prepare_one(mk, kmap, item, iord, ten, sig_eff, strikes=strikes)
    if P is None:
        return (np.nan,) * len(MC_COLS)
    mc = mc_daily_t(
        P["sigs"],
        P["corr"],
        P["beta"],
        B,
        P["strikes"],
        ten,
        n=n,
        seed=seed,
        c=c,
        pmts=pmts,
        lz_barr=lz_barr,
        lz_pmt=lz_pmt,
        dev=dev,
    )
    return (mc, *P["audit"], P["rkrw"], P["k"])
