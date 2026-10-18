# -*- coding: utf-8 -*-
"""KRW 할인커브 — CD 91일 + IRS par 금리 부트스트랩 (현행 방법론의 할인 입력).

module/features.py 의 Nelson-Siegel(krw_beta/zero_curve)은 1~3년 구간을 ~80bp 과대평가한다
(call-CD 스프레드 때문). 현행 방법론은 CD 91일을 단기 끝으로 놓고 IRS par 금리를
분기 고정(ALPHA=0.25)으로 부트스트랩한 제로커브를 쓴다.

종전에는 scratch/mc_boot_curve.py 에만 있어 추적 파이프라인에서 쓸 수 없었다.
mc_engine.mc_daily_t 의 beta 인자가 콜러블을 받으므로 curve_dz(...) 를 그대로 넘기면 된다.

인과성: market_inputs(dt) 는 발행일 dt **이전**에 공시된 값만 본다.
산출물 아님 — 원천(data/raw/rates/ 의 irs_bondweb.parquet, krw_curve.parquet)을 읽기만 한다.
"""
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import brentq

from util import file_manager as fm

SW_T = np.array([1, 2, 3, 4, 5, 7, 10, 12, 15, 20], float)
SW_C = [f"{int(t)}Y" for t in SW_T]
ALPHA = 0.25  # IRS 고정지급 주기 (분기)
T_CD = 91 / 365  # CD 91일
IRS_STALE_DAYS = 14  # 이보다 오래된 IRS 호가는 쓰지 않는다

_IRS = None
_CD_M = None


def _load() -> tuple[pd.DataFrame, pd.Series]:
    """원천 금리 파일을 지연 적재한다. 테스트가 경로를 monkeypatch 할 수 있도록 호출 시점에 읽는다.

    입력
      없음
    출력
      (irs, cd_m) — irs : DataFrame, 날짜 인덱스 × SW_C(1Y~20Y) par 금리 [%]
                    cd_m: Series, 월초 인덱스의 CD 91일 월평균 [%]
                    둘 다 모듈 전역에 캐시되어 **같은 객체가 반복 반환된다**
    """
    global _IRS, _CD_M
    if _IRS is None:
        _IRS = pd.read_parquet(fm.rate("irs_bondweb.parquet"))
        _CD_M = pd.read_parquet(fm.rate("krw_curve.parquet"))["m3"].dropna()
    return _IRS, _CD_M


def market_inputs(dt: pd.Timestamp) -> dict[str, Any] | None:
    """발행일 dt 에 알려진 CD·IRS 만 반환. 쓸 수 없으면 None.

    IRS 는 14일 이내 최신 호가까지 허용한다 (2017 추석 10일 연휴처럼 시장이 닫힌 기간에는
    마지막 호가가 그 시점의 최신 정보다). CD 는 월평균이라 그 달이 끝나기 전 값은 쓰지 않는다.

    입력
      dt : Timestamp — 발행일. **이 날짜 이전**에 공시된 값만 본다 (인과성)
    출력
      dict | None — irs_date(쓴 호가 날짜) · swaps(ndarray(10,), 소수) ·
                    cd_month(쓴 CD 월) · cd(float, 소수).
                    쓸 수 있는 입력이 없거나 IRS 가 IRS_STALE_DAYS 보다 오래되면 None
    """
    irs, cd_m = _load()
    w = irs.loc[irs.index < dt, SW_C].dropna(how="all")
    m = cd_m[cd_m.index < dt.to_period("M").to_timestamp()]
    if not len(w) or not len(m) or (dt - w.index[-1]).days > IRS_STALE_DAYS:
        return None
    return dict(
        irs_date=w.index[-1],
        swaps=w.iloc[-1].values / 100,
        cd_month=m.index[-1],
        cd=float(m.iloc[-1]) / 100,
    )


def bootstrap(cd: float | None, swaps: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """CD 단기 끝 + IRS par 금리를 분기 고정(ALPHA)으로 부트스트랩한다.

    입력
      cd    : float | None — CD 91일 (소수). None 이면 단기 끝 없이 1Y 스왑부터 간다
                             (0~1Y 가 평탄 선도가 된다)
      swaps : array-like(10,) — SW_T 순서의 par 금리 (소수). NaN 노드는 건너뛴다
    출력
      (kt, kl) — kt : ndarray, 노드 시점 [년]. 0.0 에서 시작한다
                 kl : ndarray, 같은 길이의 ln(할인계수)
    """
    kt, kl = [0.0], [0.0]
    if cd is not None:
        kt.append(T_CD)
        kl.append(-np.log1p(cd * T_CD))
    for T, s in zip(SW_T, swaps):
        if not np.isfinite(s):
            continue
        ti = np.arange(1, int(round(T / ALPHA)) + 1) * ALPHA

        def gap(lx: float, T: float = T, s: float = s, ti: np.ndarray = ti) -> float:
            """이 만기 스왑의 가치 오차 — brentq 가 0 으로 만드는 함수.

            T·s·ti 를 기본인수로 묶어 둔다. 늦은 바인딩이면 루프의 마지막 만기만 쓰이게 된다.

            입력
              lx : float — 시험 중인 노드의 ln(할인계수)
            출력
              float — 고정다리 현가 + 만기 할인계수 − 1. 0 이면 par 가 맞는다
            """
            lnd = np.interp(ti, kt + [T], kl + [lx])
            return s * ALPHA * np.exp(lnd).sum() + np.exp(lnd[-1]) - 1.0

        lx = brentq(gap, -0.5 * T, 0.2 * T, xtol=1e-15)  # gap 이 kt/kl 을 읽으므로 해를 먼저 구한다
        kt.append(T)
        kl.append(lx)
    return np.array(kt), np.array(kl)


def curve_dz(kt: np.ndarray, kl: np.ndarray) -> Callable[[npt.ArrayLike], np.ndarray]:
    """부트스트랩 노드를 '만기배열 → 제로금리' 콜러블로 바꾼다.

    `mc_engine.mc_daily_t` 의 beta 인자가 콜러블을 받으므로 결과를 그대로 넘기면 된다.

    입력
      kt, kl : ndarray — `bootstrap` 의 출력 (노드 시점, ln 할인계수)
    출력
      callable(times) -> ndarray — 만기 배열을 받아 연속복리 제로금리를 낸다.
                                   마지막 노드 밖은 **선도금리 평탄**으로 외삽하고,
                                   0 이하 만기는 1e-6 으로 바닥을 둔다
    """
    slope = (kl[-1] - kl[-2]) / (kt[-1] - kt[-2])

    def dz(times: npt.ArrayLike) -> np.ndarray:
        """만기 → 제로금리.

        입력
          times : scalar | array-like — 만기 [년]
        출력
          ndarray — 같은 길이의 연속복리 제로금리
        """
        t = np.maximum(np.atleast_1d(np.asarray(times, float)), 1e-6)
        lnd = np.where(t <= kt[-1], np.interp(t, kt, kl), kl[-1] + slope * (t - kt[-1]))
        return -lnd / t

    return dz


def par_rates(dz: Callable[[npt.ArrayLike], np.ndarray]) -> np.ndarray:
    """커브에서 IRS par 금리를 되계산한다 (부트스트랩 재현 검증용).

    입력한 par 금리와 이 결과가 일치하면 부트스트랩이 맞다 — 왕복 검증용이다.

    입력
      dz : callable — `curve_dz` 가 돌려준 제로금리 콜러블
    출력
      ndarray(10,) — SW_T 순서의 par 금리 (소수)
    """
    out = []
    for T in SW_T:
        ti = np.arange(1, int(round(T / ALPHA)) + 1) * ALPHA
        D = np.exp(-dz(ti) * ti)
        out.append((1 - D[-1]) / (ALPHA * D.sum()))
    return np.array(out)


def zero_from_date(dt: pd.Timestamp) -> Callable[[npt.ArrayLike], np.ndarray] | None:
    """발행일 하나로 바로 제로커브 콜러블을 얻는 지름길 (market_inputs → bootstrap → curve_dz).

    입력
      dt : Timestamp — 발행일
    출력
      callable | None — 제로금리 콜러블. 그 날짜에 쓸 시장 입력이 없으면 None
    """
    mi = market_inputs(dt)
    return None if mi is None else curve_dz(*bootstrap(mi["cd"], mi["swaps"]))
