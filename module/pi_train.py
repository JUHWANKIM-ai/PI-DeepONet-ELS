# -*- coding: utf-8 -*-
"""PI-DeepONet stage-1 앵커 학습 (물리정보 MC 이론가 근사).

train_curve 와 같은 규약: 폴드 학습 → predict(idx) 반환, save_path 주면 <path>.pt 저장.
stage-2 잔차(시장마진)는 물리 대상이 아니므로 여기서 건드리지 않는다 — 조립은 pipeline.predict_hybrid.

variant(물리항 조합):
  data_only  물리항 없음. PI 구조의 대조군(= CurveOperatorV2 와 사실상 동등)
  pde        + PDE 잔차                                    ← 릴리스 검증 선택 모델
  event      + 이벤트 페이오프 · 리자드 · 연속성
  full       + 만기 · KI 무차별 · 가치범위

가중치는 하드코딩 lambda 가 아니라 **첫 스텝에서 data loss 대비 비율로 자동 산출**해 고정한다.
물리항 스케일이 계약마다 수십배 차이나서 고정 lambda 는 폴드마다 다르게 작동했다.

curriculum: init_path 를 주면 그 체크포인트에서 시작한다(data_only → pde 순서 전제).
"""
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import time

import numpy as np
import numpy.typing as npt
import torch

from .data import to_tensor, zstats, znorm
from .networks import PIStateDeepONet
from .pi_collocation import PICollocationSampler
from .pi_physics import (
    continuation_relation,
    event_payoff,
    finite_or_raise,
    pde_residual,
    terminal_payoff,
)
from .train import _EarlyStop, _opt

# variant -> 활성 물리항. 순서대로 누적되는 curriculum 사슬이기도 하다.
COMPONENTS = {
    "data_only": (),
    "pde": ("pde",),
    "event": ("pde", "event", "lizard", "continuation"),
    "full": ("pde", "event", "lizard", "continuation", "terminal", "ki", "bound"),
}
# curriculum 선행 variant (해당 폴드의 이 체크포인트에서 초기화)
PREDECESSOR = {"pde": "data_only", "event": "pde", "full": "event"}

IPMT = 12  # D.CON 내 pmt_j 시작 인덱스
ILZB = 24  # lz_barr_j 시작
ILZP = 36  # lz_pmt_j 시작


def issuance_state(D: SimpleNamespace, idx: npt.ArrayLike) -> np.ndarray:
    """발행시점 상태벡터를 만든다 — PI 앵커의 '가격' 은 이 상태에서 평가한 값이다.

    S=1, 러닝최소=1, tau=만기, KI=(노낙인이면 1), event_pre=0, 진행률=0.
    노낙인(B≥1)에 KI=1 을 넣는 것은 MC 규약과 같다 — 만기에 항상 worst 를 받는다.

    입력
      D   : Dataset — D.CON(배리어)·D.TEN 을 본다
      idx : array-like[int] — 행 인덱스
    출력
      ndarray(len(idx), 10) float32 — **정규화 전** 원 스케일 상태.
                                      정규화는 `state_norm` 이 한다
    """
    n = len(idx)
    spot = np.ones((n, 3), "float32")
    barr = D.CON[idx, D.iBARR]
    ki = (barr >= 1.0).astype("float32")  # B>=1(노낙인)은 MC 규약상 항상 worst → KI 상태와 동일
    return np.c_[spot, spot.copy(), D.TEN[idx], ki, np.zeros(n), np.zeros(n)].astype("float32")


def state_norm(x: torch.Tensor, ten_m: float, ten_s: float) -> torch.Tensor:
    """상태벡터를 정규화한다 — 가격비는 고정 스케일, tau 만 데이터 기준 z-score.

    S·M 을 z-score 로 하지 않는 이유는 collocation 이 학습분포 밖 값을 넣기 때문이다.
    1 기준 ±0.25 라는 **고정 스케일**이라야 학습·물리 평가가 같은 좌표를 쓴다.
    KI·플래그·진행률은 이미 0~1 이라 건드리지 않는다.

    입력
      x     : Tensor(n, 10) — 원 스케일 상태
      ten_m : float — 만기 평균 (학습 폴드 기준)
      ten_s : float — 만기 표준편차
    출력
      Tensor(n, 10) — 사본 (입력은 바꾸지 않는다)
    """
    y = x.clone()
    y[:, :6] = (y[:, :6] - 1.0) / 0.25
    y[:, 6] = (y[:, 6] - ten_m) / ten_s
    return y


def _corr_matrix(D: SimpleNamespace, ci: np.ndarray, dev: str) -> torch.Tensor:
    """VC 의 쌍별 상관 3개를 3×3 행렬로 펼친다.

    입력
      D   : Dataset — D.VC[:, 3:6] = (rho12, rho13, rho23)
      ci  : array-like[int] — 행 인덱스
      dev : str — torch 장치
    출력
      Tensor(len(ci), 3, 3) — 대각 1, 대칭. PSD 보정은 하지 않는다
                              (PDE 잔차는 행렬을 그대로 쓰고 분해하지 않는다)
    """
    rho = to_tensor(D.VC[ci, 3:6], dev)
    corr = torch.eye(3, device=dev).repeat(len(ci), 1, 1)
    corr[:, 0, 1] = corr[:, 1, 0] = rho[:, 0]
    corr[:, 0, 2] = corr[:, 2, 0] = rho[:, 1]
    corr[:, 1, 2] = corr[:, 2, 1] = rho[:, 2]
    return corr


def _gather_schedule(
    raw: torch.Tensor, cit: torch.Tensor, j: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """관측 j 의 계약 조항 네 개를 계약별로 뽑는다 (elementwise 고급인덱싱).

    행마다 **다른 j** 를 집는다 — `raw[cit, j]` 의 두 인덱스가 같은 길이라 짝지어 색인된다.

    입력
      raw : Tensor(n, 51) — **정규화 전** 계약 (페이오프 계산용)
      cit : Tensor[int] — 계약 행 인덱스
      j   : Tensor[int] — 같은 길이의 관측 회차 (0-based)
    출력
      (K, pmt, lzb, lzp) — 각각 Tensor(len(cit),).
                           행사가 · 누적지급률 · 리자드배리어 · 리자드지급률
    """
    return raw[cit, j], raw[cit, IPMT + j], raw[cit, ILZB + j], raw[cit, ILZP + j]


def train_pi_curve(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    tr: np.ndarray,
    va: np.ndarray | None,
    variant: str = "pde",
    save_path: str | None = None,
    init_path: str | None = None,
    return_predict: bool = True,
    mask_state: bool = False,
) -> tuple[Callable[[npt.ArrayLike], np.ndarray], dict[str, Any]] | dict[str, Any]:
    """PI 앵커 학습. target 은 항상 D.MC(이론가). 반환 (predict, meta).

    predict(idx) -> np.ndarray  (발행시점 상태에서 평가한 MC_hat)
    meta 는 학습이력·물리 스케일·초기화 종류 등 진단용 dict (체크포인트에도 같이 저장).

    mask_state=True 는 state 10채널을 항상 0 으로 넣는다 → 그 채널이 forward 에 기여하지 못하므로
     **기능적으로 CurveOperatorV2(=DeepONet) 와 동일**하고, 다른 점은 trunk 첫 층 fan_in 이 61 이라
     Kaiming bound 가 1/sqrt(61) 로 작아진다는 것뿐이다. 이 초기화가 시드 분산을 크게 줄인다
     (실측: 최종 R² 시드 sd 0.0118 -> 0.0031, scratch/ablation_pi_state.py A1 vs A2).
     물리항은 비발행 상태를 평가해야 하므로 mask_state 와 함께 쓸 수 없다.

    입력
      D, cfg         : 공통 번들과 설정. train.batch · train.nit · networks.P · seed ·
                       physics.alpha(기본 0.02) · physics.batch(기본 48)
      tr, va         : ndarray[int] — 학습·검증 행. collocation 도 tr 만 쓴다
      variant        : "data_only" | "pde" | "event" | "full" — 활성 물리항 조합
      save_path      : str | None — "<...>.pt" 로 가중치·통계·meta 를 저장한다
      init_path      : str | None — 주면 그 체크포인트에서 시작한다 (curriculum)
      return_predict : bool — False 면 meta 만 돌려준다
      mask_state     : bool — state 10채널을 0 으로 고정 (위 설명 참고)
    출력
      (predict, meta) — return_predict=True
      meta            — return_predict=False
        predict : callable(idx) -> ndarray — 발행시점 상태에서 평가한 MC_hat
        meta    : dict — variant · components · physics_alpha · physics_batch ·
                  physics_scales · init_kind · mask_state · iterations ·
                  validation_mc_mse · seconds · history
    예외
      ValueError — 모르는 variant, 또는 mask_state 와 물리항을 함께 쓸 때
      FloatingPointError — 손실이나 기울기에 NaN/Inf 가 생길 때 (즉시 멈춘다)
    """
    if variant not in COMPONENTS:
        raise ValueError(f"알 수 없는 variant {variant!r} (가능: {sorted(COMPONENTS)})")
    components = COMPONENTS[variant]
    if mask_state and components:
        raise ValueError(
            "mask_state=True 는 물리항과 함께 쓸 수 없다 (물리는 비발행 상태를 평가한다)"
        )
    dev = D.DEV
    alpha = cfg.get("physics", {}).get("alpha", 0.02)
    pbatch = cfg.get("physics", {}).get("batch", 48)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # 정규화 통계는 fold train 에서만 (train_curve 와 동일 규약)
    um, us = zstats(D.CURVE, tr)
    vm, vs = zstats(D.VC, tr)
    cm, cs = zstats(D.CON, tr)
    U = to_tensor(znorm(D.CURVE, um, us), dev)
    VC = to_tensor(znorm(D.VC, vm, vs), dev)
    C = to_tensor(znorm(D.CON, cm, cs), dev)
    ym = float(D.MC[tr].mean())
    ysd = float(D.MC[tr].std() + 1e-8)
    Y = to_tensor((D.MC - ym) / ysd, dev)
    ten_m = float(D.TEN[tr].mean())
    ten_s = float(D.TEN[tr].std() + 1e-8)
    X0 = state_norm(to_tensor(issuance_state(D, np.arange(D.n)), dev), ten_m, ten_s)
    if mask_state:
        X0 = torch.zeros_like(X0)

    net = PIStateDeepONet(VC.shape[1], C.shape[1], cfg["networks"]["P"]).to(dev)
    init_kind = "scratch"
    if init_path:
        ck = torch.load(init_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state"])
        init_kind = f"curriculum:{Path(init_path).name}"
    opt = _opt(net, cfg)
    sampler = PICollocationSampler(D, tr, cfg["seed"])
    raw = to_tensor(D.CON, dev)  # 정규화 전 계약(페이오프 계산용)
    trt = torch.tensor(tr, device=dev)
    es = _EarlyStop(net, cfg, va, dev)
    hist = []
    scales = {}
    t0 = time.time()

    def actual(idx: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """원본 스케일 가치 — 물리항은 전부 이것을 쓴다 (페이오프가 원 단위라서다).

        입력
          idx   : Tensor[int] — 계약 행 인덱스
          state : Tensor(n, 10) — **원 스케일** 상태. 정규화는 안에서 한다
        출력
          Tensor(n,) — 원본 스케일 가치
        """
        return net.V(U[idx], VC[idx], C[idx], state_norm(state, ten_m, ten_s)) * ysd + ym

    def data_loss(idx: torch.Tensor) -> torch.Tensor:
        """라벨 손실 — 발행시점 상태(X0)에서 MC 라벨과의 MSE.

        표준화 공간에서 재므로 물리항(원 스케일)과 단위가 다르다 — 그래서 첫 스텝에서
        비율로 스케일을 맞춘다.

        입력
          idx : Tensor[int] — 행 인덱스
        출력
          Tensor() — 스칼라 손실
        """
        return ((net.V(U[idx], VC[idx], C[idx], X0[idx]) - Y[idx]) ** 2).mean()

    for it in range(cfg["train"]["nit"]):
        bb = trt[torch.randint(0, len(tr), (cfg["train"]["batch"],), device=dev)]
        loss = {"data": data_loss(bb)}

        if "pde" in components:
            ci, st = sampler.sample_interior(pbatch, dev)
            sampler.assert_train_only(ci)
            cit = torch.tensor(ci, device=dev)
            fixed = st.detach().clone()

            def value_fn(spot: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
                """스팟·tau 만 바꿔 가치를 내는 함수 — autograd 가 이 둘로 미분한다.

                나머지 상태 채널은 샘플한 값으로 고정한다 (fixed 의 사본).

                입력
                  spot : Tensor(n, 3) — requires_grad 가 켜진 스팟
                  tau  : Tensor(n, 1) — requires_grad 가 켜진 잔존만기
                출력
                  Tensor(n,) — 원본 스케일 가치
                """
                z = fixed.clone()
                z[:, :3] = spot
                z[:, 6:7] = tau
                return actual(cit, z)

            loss["pde"] = (
                pde_residual(
                    value_fn,
                    st[:, :3],
                    st[:, 6:7],
                    to_tensor(D.VC[ci, :3], dev),
                    _corr_matrix(D, ci, dev),
                    to_tensor(D.R[ci], dev),
                )
                ** 2
            ).mean()

        if any(k in components for k in ("event", "lizard", "continuation")):
            ci, j, pre, post = sampler.sample_event(pbatch, dev)
            sampler.assert_train_only(ci)
            cit = torch.tensor(ci, device=dev)
            jt = torch.tensor(j, device=dev)
            K, pmt, lzb, lzp = _gather_schedule(raw, cit, jt)
            target, active, regular, lizard = event_payoff(
                pre[:, :3], pre[:, 3:6], K, pmt, lzb, lzp
            )
            pb = actual(cit, pre)
            pa = actual(cit, post)
            zero = pb.sum() * 0
            loss["event"] = ((pb[regular] - target[regular]) ** 2).mean() if regular.any() else zero
            loss["lizard"] = ((pb[lizard] - target[lizard]) ** 2).mean() if lizard.any() else zero
            loss["continuation"] = continuation_relation(pb, pa, ~active)

        if "terminal" in components:
            ci, st = sampler.sample_terminal(pbatch, dev)
            sampler.assert_train_only(ci)
            cit = torch.tensor(ci, device=dev)
            target = terminal_payoff(st[:, :3], raw[cit, D.iKlast], raw[cit, IPMT + 11], st[:, 7])
            loss["terminal"] = ((actual(cit, st) - target) ** 2).mean()

        if "ki" in components:
            ci, s0, s1 = sampler.sample_ki_boundary(pbatch, dev)
            if len(ci):
                sampler.assert_train_only(ci)
                cit = torch.tensor(ci, device=dev)
                loss["ki"] = ((actual(cit, s0) - actual(cit, s1)) ** 2).mean()
            else:
                loss["ki"] = loss["data"] * 0

        if "bound" in components:
            ci, st = sampler.sample_interior(pbatch, dev)
            sampler.assert_train_only(ci)
            cit = torch.tensor(ci, device=dev)
            pr = actual(cit, st)
            cap = 1.0 + raw[cit, IPMT : IPMT + 12].amax(1)
            loss["bound"] = (torch.relu(-pr).square() + torch.relu(pr - cap).square()).mean()

        total = loss["data"]
        for k in components:
            if k not in scales:  # 첫 스텝에서만 스케일 확정
                scales[k] = float(loss["data"].detach() / (loss[k].detach() + 1e-12))
            total = total + alpha * scales[k] * loss[k]

        finite_or_raise("total_loss", total)
        opt.zero_grad()
        total.backward()
        for p in net.parameters():
            if p.grad is not None:
                finite_or_raise("gradient", p.grad)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

        if (it + 1) % cfg["train"].get("es_every", 100) == 0:
            hist.append(
                {
                    "iteration": it + 1,
                    "total": float(total.detach()),
                    **{k: float(v.detach()) for k, v in loss.items()},
                }
            )
        if es.step(it, lambda vi: data_loss(vi)):
            break

    es.restore()
    net.eval()
    meta = {
        "variant": variant,
        "components": list(components),
        "physics_alpha": alpha,
        "physics_batch": pbatch,
        "physics_scales": scales,
        "init_kind": init_kind,
        "mask_state": bool(mask_state),
        "iterations": hist[-1]["iteration"] if hist else cfg["train"]["nit"],
        "validation_mc_mse": es.best if es.on else None,
        "seconds": time.time() - t0,
        "history": hist,
    }
    if save_path:
        p = save_path if str(save_path).endswith(".pt") else str(save_path) + ".pt"
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": net.state_dict(),
                "P": cfg["networks"]["P"],
                "nvc": VC.shape[1],
                "ncon": C.shape[1],
                "um": um,
                "us": us,
                "vm": vm,
                "vs": vs,
                "cm": cm,
                "cs": cs,
                "ym": ym,
                "ysd": ysd,
                "ten_m": ten_m,
                "ten_s": ten_s,
                "train_indices": np.asarray(tr),
                **meta,
            },
            p,
        )

    def predict(idx: npt.ArrayLike) -> np.ndarray:
        """발행시점 상태에서 MC_hat 을 낸다 — 기존 앵커와 같은 인터페이스.

        입력
          idx : array-like[int] — 행 인덱스
        출력
          ndarray — 길이 len(idx), 원본 스케일
        """
        with torch.no_grad():
            ix = torch.tensor(np.asarray(idx), device=dev)
            st = to_tensor(issuance_state(D, np.asarray(idx)), dev)
            if mask_state:
                st = torch.zeros_like(st)
                return (net.V(U[ix], VC[ix], C[ix], st) * ysd + ym).cpu().numpy()
            return actual(ix, st).cpu().numpy()

    return (predict, meta) if return_predict else meta


def load_pi_predictor(D: SimpleNamespace, path: str) -> Callable[[npt.ArrayLike], np.ndarray]:
    """저장된 PI 앵커 가중치를 행 인덱스 예측기로 되살린다 (forward 만, 재학습 없음).

    mask_state 는 체크포인트에 기록된 값을 따른다 — 학습과 추론 경로가 어긋나면 안 된다.

    입력
      D    : Dataset — 입력 세 블록과 발행시점 상태를 만든다
      path : str — .pt 경로
    출력
      callable(idx) -> ndarray — 원본 스케일 MC_hat 예측기
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PIStateDeepONet(ck["nvc"], ck["ncon"], ck["P"]).to(D.DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    U = to_tensor(znorm(D.CURVE, ck["um"], ck["us"]), D.DEV)
    VC = to_tensor(znorm(D.VC, ck["vm"], ck["vs"]), D.DEV)
    C = to_tensor(znorm(D.CON, ck["cm"], ck["cs"]), D.DEV)
    masked = bool(ck.get("mask_state", False))

    def predict(idx: npt.ArrayLike) -> np.ndarray:
        """복원된 망으로 발행시점 MC_hat 을 낸다.

        입력
          idx : array-like[int] — 행 인덱스
        출력
          ndarray — 길이 len(idx), 원본 스케일
        """
        with torch.no_grad():
            ix = torch.tensor(np.asarray(idx), device=D.DEV)
            st = state_norm(
                to_tensor(issuance_state(D, np.asarray(idx)), D.DEV), ck["ten_m"], ck["ten_s"]
            )
            if masked:
                st = torch.zeros_like(st)
            return (net.V(U[ix], VC[ix], C[ix], st) * ck["ysd"] + ck["ym"]).cpu().numpy()

    return predict


def lizard_no_ki_mask(D: SimpleNamespace) -> np.ndarray:
    """LIZARD no-KI 라우팅 키 — **계약 조건만으로** 유도(parquet 라벨 불필요, 누출 없음).
    LIZARD = 리자드 배리어가 중립(1.2)이 아닌 회차가 하나라도 있음 / no-KI = 낙인배리어 B >= 1.
    (parquet 의 opt_type·ki_yn 과 58,790건 전부 일치 확인됨)

    입력
      D : Dataset — D.CON 의 리자드 배리어 12칸과 D.iBARR 만 본다
    출력
      ndarray(n,) bool — True 면 PI(PDE) 앵커로 라우팅할 행
    """
    is_lizard = (D.CON[:, ILZB : ILZB + 12] < 1.15).any(1)
    no_ki = D.CON[:, D.iBARR] >= 1.0
    return is_lizard & no_ki


def physics_diagnostics(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    path: str,
    tr: np.ndarray,
    n: int = 256,
    seed: int = 0,
) -> dict[str, float]:
    """저장된 앵커의 **물리 위반량**을 fold-train collocation 에서 측정 (variant 무관, 동일 조건 비교용).

    물리항을 안 켠 모델(data_only)에도 그대로 적용해 '물리가 실제로 좋아졌는지' 를 본다.
    (릴리스는 physics_violations.csv 만 배포하고 생성 코드는 동봉하지 않아 여기서 새로 구현)

    입력
      D, cfg : 공통 번들과 설정 (cfg 는 현재 호출 규약을 맞추는 자리다)
      path   : str — 평가할 체크포인트 .pt
      tr     : ndarray[int] — collocation 을 뽑을 학습 행
      n      : int — 항목당 표본 수
      seed   : int — 추첨 시드. **같은 seed 면 같은 collocation** 이라 모델 간 비교가 짝지어진다
    출력
      dict — pde · terminal · event · lizard · continuation · ki 각 항의 제곱오차 평균.
             해당 표본이 없는 항은 0.0 이다
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PIStateDeepONet(ck["nvc"], ck["ncon"], ck["P"]).to(D.DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    dev = D.DEV
    U = to_tensor(znorm(D.CURVE, ck["um"], ck["us"]), dev)
    VC = to_tensor(znorm(D.VC, ck["vm"], ck["vs"]), dev)
    C = to_tensor(znorm(D.CON, ck["cm"], ck["cs"]), dev)
    raw = to_tensor(D.CON, dev)
    ten_m, ten_s, ym, ysd = ck["ten_m"], ck["ten_s"], ck["ym"], ck["ysd"]

    def actual(idx: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """원본 스케일 가치 — `train_pi_curve` 의 같은 이름 함수와 같은 정의다.

        입력
          idx   : Tensor[int] — 계약 행 인덱스
          state : Tensor(n, 10) — 원 스케일 상태
        출력
          Tensor(n,) — 원본 스케일 가치
        """
        return net.V(U[idx], VC[idx], C[idx], state_norm(state, ten_m, ten_s)) * ysd + ym

    s = PICollocationSampler(D, tr, seed)
    out = {}

    ci, st = s.sample_interior(n, dev)
    cit = torch.tensor(ci, device=dev)
    fixed = st.detach().clone()

    def value_fn(spot: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """스팟·tau 만 바꿔 가치를 내는 함수 (PDE 잔차 측정용).

        입력
          spot : Tensor(n, 3) — requires_grad 가 켜진 스팟
          tau  : Tensor(n, 1) — requires_grad 가 켜진 잔존만기
        출력
          Tensor(n,) — 원본 스케일 가치
        """
        z = fixed.clone()
        z[:, :3] = spot
        z[:, 6:7] = tau
        return actual(cit, z)

    res = pde_residual(
        value_fn,
        st[:, :3],
        st[:, 6:7],
        to_tensor(D.VC[ci, :3], dev),
        _corr_matrix(D, ci, dev),
        to_tensor(D.R[ci], dev),
    )
    out["pde"] = float((res**2).mean().detach())

    with torch.no_grad():
        ci, st = s.sample_terminal(n, dev)
        cit = torch.tensor(ci, device=dev)
        tgt = terminal_payoff(st[:, :3], raw[cit, D.iKlast], raw[cit, IPMT + 11], st[:, 7])
        out["terminal"] = float(((actual(cit, st) - tgt) ** 2).mean())

        ci, j, pre, post = s.sample_event(n, dev)
        cit = torch.tensor(ci, device=dev)
        jt = torch.tensor(j, device=dev)
        K, pmt, lzb, lzp = _gather_schedule(raw, cit, jt)
        tgt, active, regular, lizard = event_payoff(pre[:, :3], pre[:, 3:6], K, pmt, lzb, lzp)
        pb = actual(cit, pre)
        pa = actual(cit, post)
        out["event"] = float(((pb[regular] - tgt[regular]) ** 2).mean()) if regular.any() else 0.0
        out["lizard"] = float(((pb[lizard] - tgt[lizard]) ** 2).mean()) if lizard.any() else 0.0
        out["continuation"] = float(continuation_relation(pb, pa, ~active))

        ci, s0, s1 = s.sample_ki_boundary(n, dev)
        if len(ci):
            cit = torch.tensor(ci, device=dev)
            out["ki"] = float(((actual(cit, s0) - actual(cit, s1)) ** 2).mean())
        else:
            out["ki"] = 0.0
    return out
