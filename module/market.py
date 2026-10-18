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
    """els3_dataset (발행일 정렬). 캐시된다."""
    global _D
    if _D is None:
        _D = (
            pd.read_parquet(fm.DATA / "els3_dataset.parquet")
            .sort_values("isu_ord")
            .reset_index(drop=True)
        )
    return _D


def returns():
    """티커 -> 로그수익률 Series. module.mc.load_market 의 RET 과 같다."""
    from .mc import load_market

    return load_market()["RET"]


def calendar(ret=None):
    """KRX 거래일 달력 (^KS200 수익률 인덱스)."""
    r = (ret or returns())["^KS200"]
    return r.dropna().index


def ns_zero(dt, krw_file="krw_curve_fred.parquet"):
    """FRED 3노드에 Nelson-Siegel 을 맞춘 제로커브 콜러블 (변형 A 비교용). 결측이면 None."""
    K = pd.read_parquet(fm.rate(krw_file))
    krow = K.asof(dt).values
    if np.isnan(krow).any():
        return None
    x = NS_TAUS / NS_LAM
    e = np.exp(-x)
    A = np.stack([np.ones(3), (1 - e) / x, (1 - e) / x - e], 1)
    b, *_ = np.linalg.lstsq(A, krow, rcond=None)

    def dz(times, b=b):
        t = np.maximum(np.atleast_1d(times), 1e-6)
        xx = t / NS_LAM
        ee = np.exp(-xx)
        return (np.stack([np.ones_like(t), (1 - ee) / xx, (1 - ee) / xx - ee], 1) @ b) / 100.0

    return dz


def schedule(row):
    """계약 스케줄 (행사가, 누적지급률, 리자드배리어, 리자드지급률) — 유효 회차만."""
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

    반환: dt, tks, sig(3), corr(3x3), dz(콜러블), dkw, divlog(Nx3 또는 None), qcont(3), ndiv, div.

    **dkw 를 쓸 것.** mc_engine 에 그대로 풀어 넘길 배당 kwargs 다. divlog 를 직접 꺼내면
    lseg(기본값)에서 None 이 넘어가 배당이 조용히 사라진다 — build_nb_viz 가 실제로 그랬다."""
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
