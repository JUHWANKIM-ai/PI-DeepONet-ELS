# -*- coding: utf-8 -*-
"""상품 하나의 '발행일 기준 시장 상태' 를 한 곳에서 조립한다.

종전에는 같은 조립이 module/mc.py · module/build_source.py · scratch/mc_discrete_div.py ·
5_MC_implied_vol.ipynb 네 곳에 흩어져 있었고, 서로 다른 커브 파일(krw_curve vs krw_curve_fred)과
서로 다른 상관 정의를 쓰고 있었다. 현행 방법론의 정의는 여기 하나다.

  변동성  features.vol_ewma_lam  (6개월 EWMA, λ=0.99)
  상관    features.corr_bd       (KRX 영업일 120일, 쌍별)
  커브    curve.zero_from_date   (CD 91일 + IRS par 부트스트랩)
  배당    dividend.build_div     (이산, 지수는 추종 ETF 프록시)

NS 커브(변형 A 비교용)는 ns_zero 로 따로 둔다 — 드리프트·할인에 쓰는 대체 커브다.
전부 발행일 **이전** 자료만 본다.
"""
import numpy as np
import pandas as pd

from util import file_manager as fm
from . import curve as CV
from . import dividend as DV
from . import features as F

NS_TAUS = np.array([0.08, 0.25, 10.0])  # FRED 커브의 3개 노드 (call · 3M · 10Y)
NS_LAM = 1.5
_D = None


def dataset():
    """els3_dataset 을 발행일 순으로 읽어 돌려준다. 첫 호출에서만 읽고 전역에 캐시한다.

    입력
      없음
    출력
      DataFrame — 3스타 상품 전수. isu_ord 오름차순, 인덱스 재설정.
                  **같은 객체를 매번 돌려준다** — 호출부가 수정하면 다른 호출부가 본다
    """
    global _D
    if _D is None:
        _D = (
            pd.read_parquet(fm.DATA / "els3_dataset.parquet")
            .sort_values("isu_ord")
            .reset_index(drop=True)
        )
    return _D


def returns():
    """티커 -> 로그수익률 Series. `module.mc.load_market` 의 RET 과 같은 객체다.

    입력
      없음
    출력
      dict[str -> Series] — 티커에서 일별 로그수익률로.
                            가격 적재 경로가 하나여야 하므로 mc 쪽을 그대로 쓴다
    """
    from .mc import load_market

    return load_market()["RET"]


def calendar(ret=None):
    """KRX 거래일 달력 (^KS200 수익률의 인덱스).

    입력
      ret : dict | None — `returns()` 의 결과. None 이면 여기서 부른다
            (반복 호출에서는 미리 받아 넘기는 게 싸다)
    출력
      DatetimeIndex — NaN 을 뺀 ^KS200 거래일. `features.corr_bd` 의 창 기준이 된다
    """
    r = (ret or returns())["^KS200"]
    return r.dropna().index


def ns_zero(dt, krw_file="krw_curve_fred.parquet"):
    """FRED 3노드에 Nelson-Siegel 을 맞춘 제로커브 콜러블 (변형 A 비교용).

    정본(변형 B)은 `curve.zero_from_date` 의 CD+IRS 부트스트랩이다. 이쪽 NS 는 1~3년을
    ~80bp 과대평가한다 — 비교 대상으로만 쓴다.

    입력
      dt       : Timestamp — 발행일. asof 로 그 시점 값을 집는다
      krw_file : str — data/raw/rates/ 안의 파일명
    출력
      callable(times) -> ndarray | None — 제로금리 콜러블 (소수).
                                          세 노드 중 하나라도 결측이면 None
    """
    K = pd.read_parquet(fm.rate(krw_file))
    krow = K.asof(dt).values
    if np.isnan(krow).any():
        return None
    x = NS_TAUS / NS_LAM
    e = np.exp(-x)
    A = np.stack([np.ones(3), (1 - e) / x, (1 - e) / x - e], 1)
    b, *_ = np.linalg.lstsq(A, krow, rcond=None)

    def dz(times, b=b):
        """만기 → NS 제로금리. b 를 기본인수로 묶어 적합 결과를 고정한다.

        입력
          times : scalar | array-like — 만기 [년]
        출력
          ndarray — 같은 길이의 제로금리 (소수)
        """
        t = np.maximum(np.atleast_1d(times), 1e-6)
        xx = t / NS_LAM
        ee = np.exp(-xx)
        return (np.stack([np.ones_like(t), (1 - ee) / xx, (1 - ee) / xx - ee], 1) @ b) / 100.0

    return dz


def schedule(row):
    """계약 스케줄을 유효 회차 길이로 잘라 네 리스트로 돌려준다.

    입력
      row : Series — 상품 한 행. nobs 와 strk_/pmt_/lz_barr_/lz_pmt_ 열을 본다
    출력
      (strk, pmt, lz_barr, lz_pmt) — 각각 길이 nobs 의 list[float].
                                     12칸 패딩을 잘라낸 것이므로 엔진에 그대로 넘긴다
    """
    k = int(row["nobs"])
    return (
        [float(row[f"strk_{j}"]) for j in range(k)],
        [float(row[f"pmt_{j}"]) for j in range(k)],
        [float(row[f"lz_barr_{j}"]) for j in range(k)],
        [float(row[f"lz_pmt_{j}"]) for j in range(k)],
    )


def state_for(row, ret=None, cal=None, div="lseg"):
    """**현행 방법론**의 발행일 시장상태. 하나라도 못 만들면 None.

    div: 배당 소스
      "lseg"  LSEG 배당수익률 -> 연속 q               (qcont 를 엔진에 넘긴다) — **기본값**
      "yf"    yfinance 배당 이벤트 -> 이산배당 divlog  (divlog 를 엔진에 넘긴다) — 비교용
      원천 전용 재현은 share_underlyings_v2 만 쓰는데 거기에는 배당락일·금액이 없어
      이산배당을 만들 수 없다. 그래서 기본값이 연속 q 다.
      LSEG 에는 배당락일·금액이 없어 이산배당을 만들 수 없다. 자세한 이유는 module/dividend.py 참고.

    **dkw 를 쓸 것.** mc_engine 에 그대로 풀어 넘길 배당 kwargs 다. divlog 를 직접 꺼내면
    lseg(기본값)에서 None 이 넘어가 배당이 조용히 사라진다 — build_nb_viz 가 실제로 그랬다.

    입력
      row : Series — 상품 한 행. isu_ord · udl1~3 · tenor 를 본다
      ret : dict | None — `returns()`. None 이면 여기서 부른다
      cal : DatetimeIndex | None — KRX 달력. None 이면 여기서 만든다
      div : "lseg" | "yf" — 위 설명 참고
    출력
      dict | None — dt · tks · sig(3) · corr(3,3) · dz(콜러블) · dkw ·
                    divlog(N,3 또는 None) · qcont(3) · ndiv · div.
        네 입력(수익률 계열 · 변동성 · 상관 · 커브 · 배당) 중 **하나라도** 못 만들면 None →
        호출부가 그 상품을 탈락시킨다
    """
    ret = ret if ret is not None else returns()
    cal = cal if cal is not None else calendar(ret)
    dt = pd.Timestamp.fromordinal(int(row["isu_ord"]))
    tks = [row["udl1"], row["udl2"], row["udl3"]]
    rets = [ret.get(t) for t in tks]
    if any(r is None for r in rets):
        return None
    sig = [F.vol_ewma_lam(r, dt) for r in rets]
    corr, _ = F.corr_bd(rets, dt, cal)
    dz = CV.zero_from_date(dt)
    if corr is None or dz is None or np.isnan(sig).any():
        return None
    got = DV.div_kwargs(tks, dt, float(row["tenor"]), div)  # 소스 분기는 module.dividend 한 곳
    if got is None:
        return None
    dkw, qc, nev = got
    return dict(
        dt=dt,
        tks=tks,
        sig=sig,
        corr=corr,
        dz=dz,
        dkw=dkw,
        divlog=dkw.get("divlog"),
        qcont=qc,
        ndiv=nev,
        div=div,
    )
