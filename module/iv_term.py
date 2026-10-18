# -*- coding: utf-8 -*-
"""ATM 내재변동성 + 차용 기간구조 -> ELS 만기 변동성 σ(T).

보유 자료는 기초자산별 **단일 만기 ATM IV** 뿐이다 (data/raw/iv_daily_atm, 만기·행사가 차원 없음).
그래서 기간구조는 공표 변동성지수에서 **형태만** 빌려온다:

    σ_udl(T) = ATM_udl · σ_VIX(T) / σ_VIX(T_ref)

σ_VIX(T) 는 같은 날의 VIX9D·VIX·VIX3M·VIX6M(9·30·93·182일)으로 만든 **총분산 곡선**에서 읽는다.

    w(T) = σ²(T)·T,   σ(T) = sqrt(w(T) / T)

- 노드 사이는 w 에 대해 **선형보간** — IV 자체를 보간하면 캘린더 차익 부재가 깨지므로 반드시 w 축에서.
- 노드는 w 가 비감소가 되도록 누적최댓값으로 정리한다 (역전 구간의 캘린더 차익 제거).
- 최장 노드(182일) 이후는 마지막 구간 기울기로 **선형 외삽** = 6개월 이후 선도분산 일정.
  ELS 만기(대부분 3년)는 항상 이 외삽 구간이다.

한계(자문 시 확인 필요):
  - 미국 S&P 500 의 기간구조 형태를 유로·홍콩·한국·일본 지수에 그대로 차용한다.
  - 6개월 이후는 시장 자료가 아니라 외삽이다.
  - ATM 이라 하방 스큐는 담지 못한다.
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd

from util import file_manager as fm

T_REF_DAYS = 30  # 보유 ATM IV 의 만기 가정 (1개월, 관례)
VIX_NODES = [("VIX9D", 9), ("VIX", 30), ("VIX3M", 93), ("VIX6M", 182)]

# ELS 기초자산(야후식 티커) -> ATM IV 원천 notion_ric.
#   ^STOXX50E 는 원천에 EURO STOXX 50(.STOXX50E) 이 없어 STOXX Europe 50(.STOXX50) 으로 대용한다.
#   다른 지수(영국·스위스 방어주 포함)라 IV 가 HV 보다 평균 2.8%p 낮다 -> variant 로 분리해 다룬다.
IVMAP = {
    "^KS200": ".KS200",
    "^GSPC": ".SPX",
    "^HSCE": ".HSCE",
    "^HSI": ".HSI",
    "^N225": ".N225",
    "^NDX": ".NDX",
    "^GDAXI": ".GDAXI",
    "^STOXX50E": ".STOXX50",
}
PROXY_SUSPECT = {"^STOXX50E"}


def load_atm_iv(root=None):
    """원천 ATM IV CSV 전부를 읽어 자산별 계열로 묶는다.

    입력
      root : str | Path | None — 저장소 루트. None 이면 fm.ROOT (테스트가 바꿀 수 있다)
    출력
      dict[str -> Series] — notion_ric 에서 일별 IV 로 (**소수**, 0.18 = 18%).
                            날짜 중복은 마지막 값을 쓴다
    예외
      FileNotFoundError — iv_daily_atm CSV 가 없을 때 (zip 해제가 필요하다)
    """
    root = Path(root) if root else fm.ROOT
    files = sorted(glob.glob(str(root / "data/raw/iv_daily_atm/iv_daily_atm/*.csv")))
    if not files:
        raise FileNotFoundError("data/raw/iv_daily_atm/iv_daily_atm/*.csv 가 없다 (zip 해제 필요)")
    big = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    big["calc_date"] = pd.to_datetime(big["calc_date"])
    out = {}
    for ric, g in big.groupby("notion_ric"):
        s = g.set_index("calc_date")["iv"].sort_index()
        out[ric] = s[~s.index.duplicated(keep="last")].astype(float)
    return out


def load_vix_term(root=None):
    """VIX 기간구조 네 노드를 하나의 패널로 묶는다.

    입력
      root : str | Path | None — 저장소 루트. None 이면 fm.ROOT
    출력
      DataFrame — 날짜 인덱스 × VIX_NODES 라벨(VIX9D · VIX · VIX3M · VIX6M),
                  값은 **소수** 변동성 (원천 지수값을 /100)
    """
    root = Path(root) if root else fm.ROOT
    cols = {}
    for lab, _ in VIX_NODES:
        v = pd.read_parquet(root / "data/cache/volidx" / f"{lab}.parquet").iloc[:, 0]
        v.index = pd.to_datetime(v.index)
        cols[lab] = v[~v.index.duplicated(keep="last")].astype(float) / 100.0
    return pd.DataFrame(cols).sort_index()


def total_variance_curve(sig_nodes, days_nodes):
    """노드 (σ, 일수) 를 총분산 곡선 (T, w=σ²T) 으로 바꾼다.

    w 를 누적최댓값으로 눌러 **비감소**를 강제한다 — 역전 구간을 그대로 두면 캘린더 차익이
    생긴다.

    입력
      sig_nodes  : array-like — 노드별 변동성 (소수). 비유한·0 이하는 버린다
      days_nodes : array-like — 같은 순서의 만기 [일]
    출력
      (t, w) | None — t : ndarray, 만기 [년] 오름차순
                      w : ndarray, 같은 길이의 비감소 총분산
                      쓸 수 있는 노드가 2개 미만이면 None
    """
    s = np.asarray(sig_nodes, float)
    t = np.asarray(days_nodes, float) / 365.0
    ok = np.isfinite(s) & (s > 0)
    s, t = s[ok], t[ok]
    if len(t) < 2:
        return None
    o = np.argsort(t)
    t, s = t[o], s[o]
    w = np.maximum.accumulate(s**2 * t)  # 비감소 강제
    return t, w


def sigma_at(t, w, T):
    """총분산 곡선에서 만기 T 의 변동성을 읽는다.

    보간은 **w 축에서** 한다 — IV 자체를 보간하면 캘린더 차익 부재가 깨진다.
    ELS 만기(대부분 3년)는 항상 외삽 구간이다.

    입력
      t : ndarray — 노드 만기 [년] 오름차순
      w : ndarray — 같은 길이의 총분산
      T : float — 읽을 만기 [년]
    출력
      float — 변동성 (소수).
        T 가 최단 노드보다 짧으면 그 노드 값, 노드 범위 안이면 선형보간,
        최장 노드 밖이면 마지막 구간 기울기로 선형 외삽(선도분산 일정, 음수 불가)
    """
    if T <= t[0]:
        return float(np.sqrt(w[0] / t[0]))  # 최단 노드보다 짧으면 그 노드의 변동성
    if T <= t[-1]:
        return float(np.sqrt(np.interp(T, t, w) / T))
    slope = max((w[-1] - w[-2]) / (t[-1] - t[-2]), 0.0)  # 선도분산 (음수 불가)
    return float(np.sqrt((w[-1] + slope * (T - t[-1])) / T))


def term_ratio(vix_row, T, t_ref_days=T_REF_DAYS):
    """σ_VIX(T) / σ_VIX(T_ref) — 자산별 ATM IV 에 곱할 기간구조 **배수**.

    비율로 쓰기 때문에 VIX 의 절대 수준은 상쇄되고 형태만 빌려온다.

    입력
      vix_row     : array-like(4,) — VIX_NODES 순서의 소수 변동성 (그 날짜 한 줄)
      T           : float — 목표 만기 [년]
      t_ref_days  : int — 기준 만기 [일]. 기본 T_REF_DAYS(30) = 보유 ATM IV 의 만기 가정
    출력
      float — 배수. 곡선을 만들 수 없으면 np.nan
    """
    cur = total_variance_curve(vix_row, [d for _, d in VIX_NODES])
    if cur is None:
        return np.nan
    t, w = cur
    return sigma_at(t, w, T) / sigma_at(t, w, t_ref_days / 365.0)


def asof(series, dt):
    """dt 이하의 마지막 관측값 (미래정보 차단).

    `market.state_for` 의 다른 입력들이 dt **미만**을 쓰는 것과 달리 여기는 dt **이하**다 —
    IV 는 당일 종가 지표로 쓰므로 그 날짜를 포함한다.

    입력
      series : Series — 날짜 인덱스 계열
      dt     : Timestamp — 기준일
    출력
      float — 마지막 관측값. dt 이하 관측이 없으면 np.nan
    """
    s = series[series.index <= dt]
    return float(s.iloc[-1]) if len(s) else np.nan
