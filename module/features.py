# -*- coding: utf-8 -*-
"""공용 피처 유틸 (DRY) — KRW Nelson-Siegel 곡선 + 180일 역사 변동성·상관.
0_data(데이터셋 branch 피처 재유도)와 MC 재산출 노트북이 공유한다.
곡선/변동성 정의는 데이터셋을 만든 프라이서와 동일하게 유지(재현성)."""
import numpy as np
import pandas as pd

LAM = 1.5
KR_TAUS = np.array([0.08, 0.25, 10.0])  # 콜·3M 은행간·10Y 국채 (FRED)
MSAMP = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10], dtype=float)  # branch 곡선 노드 u0..u9
VOL_WIN = 180
CORR_WIN = 180
EWMA_SPAN = 120  # 자문안 변동성: λ = 1 − 2/(span+1) ≈ 0.9835 (반감기 ≈ 42 거래일)
ADV_CORR_WIN = 120  # 6개월 = 120 영업일 (현행 방법론의 변동성·상관 창)
EWMA_LAM = 0.99  # 현행 방법론의 EWMA 감쇠 (반감기 ≈ 69 거래일). 정본 mc 가 이 값이다


def nsb(t):
    """Nelson-Siegel 기저 [level, slope, curvature] (LAM 고정).

    입력
      t : scalar | array-like — 만기 [년]
    출력
      ndarray(len(t), 3) — 각 행이 그 만기의 세 기저값
    """
    t = np.atleast_1d(np.asarray(t, float))
    x = t / LAM
    e = np.exp(-x)
    return np.stack([np.ones_like(t), (1 - e) / x, (1 - e) / x - e], 1)


def zero_curve(beta, times):
    """NS 제로금리 곡선 = nsb(times) @ beta / 100.

    입력
      beta  : ndarray(3,) — [level, slope, curvature], 단위 %
      times : scalar | array-like — 만기 [년]. 0 이하는 1e-6 으로 바닥을 둔다
    출력
      ndarray — 같은 길이의 제로금리 (**소수**, 퍼센트가 아니다)
    """
    return (nsb(np.maximum(np.atleast_1d(times), 1e-6)) @ beta) / 100.0


def krw_beta(row):
    """세 금리 관측점에 NS 를 최소제곱 적합한다.

    입력
      row : array-like(3,) — [call, m3, y10], 발행일 asof 값 [%].
            만기는 KR_TAUS(0.08 · 0.25 · 10.0) 로 고정이다
    출력
      ndarray(3,) | None — NS beta [%]. 하나라도 NaN 이면 None →
                           호출부가 그 상품을 탈락시킨다
    """
    row = np.asarray(row, float)
    if np.isnan(row).any():
        return None
    beta, *_ = np.linalg.lstsq(nsb(KR_TAUS), row, rcond=None)
    return beta


def krw_curve_nodes(beta):
    """branch 곡선 노드 u0..u9 = MSAMP 만기점의 KRW NS 제로금리.

    입력
      beta : ndarray(3,) — `krw_beta` 의 출력
    출력
      ndarray(10,) float32 — MSAMP(0.25 ~ 10년) 10개 노드의 제로금리 (소수).
                             DeepONet branch 의 입력 블록이다
    """
    return zero_curve(beta, MSAMP).astype(np.float32)


def chol_psd(corr):
    """촐레스키 분해. 양정부호가 아니면 고유값을 깎아 보정한 뒤 다시 분해한다.

    쌍별 상관(`corr_bd`)은 각 쌍이 다른 날짜 집합을 쓰므로 PSD 가 보장되지 않는다 —
    이 보정이 그래서 필요하다.

    입력
      corr : ndarray(k, k) — 상관행렬
    출력
      ndarray(k, k) — 하삼각 촐레스키 인자. 보정한 경우 대각을 1 로 재정규화한 뒤 분해한다
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(corr)
        ev = np.clip(ev, 1e-8, None)
        c2 = V @ np.diag(ev) @ V.T
        d = np.sqrt(np.diag(c2))
        c2 = c2 / np.outer(d, d)
        return np.linalg.cholesky(c2)


def vol180(ret, dt):
    """발행 전 180 거래일 로그수익률 연율 변동성 (std × √252).

    입력
      ret : Series — 일별 로그수익률, DatetimeIndex
      dt  : Timestamp — 발행일. **이 날짜 이전**만 쓴다 (인과성)
    출력
      float — 연율 변동성. 표본이 60 미만이면 np.nan
    """
    w = ret[ret.index < dt].tail(VOL_WIN)
    return float(w.std() * np.sqrt(252)) if len(w) >= 60 else np.nan


def vol_win(ret, dt, win):
    """발행 전 win 거래일 **단순** 역사변동성 (std × √252).

    `vol180` 은 win 이 180 으로 고정된 특수화다 — MC 변형 A 는 win=120 을 쓴다.

    입력
      ret : Series — 일별 로그수익률
      dt  : Timestamp — 발행일
      win : int — 창 길이 [거래일]
    출력
      float — 연율 변동성. 표본이 60 미만이면 np.nan
    """
    w = ret[ret.index < dt].tail(win)
    return float(w.std() * np.sqrt(252)) if len(w) >= 60 else np.nan


def vol_ewma(ret, dt, span=EWMA_SPAN):
    """발행 전 span 거래일 EWMA 연율 변동성 — ewm(span) 가중 표준편차의 마지막 값 × √252.

    가중치 (1−α)^i 는 창 안에서 재정규화된다(adjust=True).
    주의: 이건 **자문안** 정의(λ≈0.9835)다. 현행 방법론은 `vol_ewma_lam`(λ=0.99) 을 쓴다.

    입력
      ret  : Series — 일별 로그수익률
      dt   : Timestamp — 발행일
      span : int — EWMA span. 창 길이로도 함께 쓰인다
    출력
      float — 연율 변동성. 표본이 60 미만이면 np.nan
    """
    w = ret[ret.index < dt].tail(span)
    return (
        float(w.ewm(span=span, adjust=True).std().iloc[-1] * np.sqrt(252))
        if len(w) >= 60
        else np.nan
    )


def vol_ewma_lam(ret, dt, lam=EWMA_LAM, win=ADV_CORR_WIN):
    """**현행 방법론의 변동성** — 발행 전 win 거래일(=6개월) EWMA, 감쇠 λ 를 직접 지정.

    `vol_ewma` 의 span 표기와 달리 λ 를 그대로 받는다 (λ=0.99 → α=0.01).
    정본 mc 열이 이 정의로 산출됐다.

    입력
      ret : Series — 일별 로그수익률
      dt  : Timestamp — 발행일
      lam : float — EWMA 감쇠. 기본 EWMA_LAM(0.99), 반감기 ≈ 69 거래일
      win : int — 창 길이 [거래일]. 기본 ADV_CORR_WIN(120)
    출력
      float — 연율 변동성. 표본이 60 미만이면 np.nan
    """
    w = ret[ret.index < dt].tail(win)
    return (
        float(w.ewm(alpha=1 - lam, adjust=True).std().iloc[-1] * np.sqrt(252))
        if len(w) >= 60
        else np.nan
    )


def corr_bd(rets, dt, calendar, win=ADV_CORR_WIN):
    """**현행 방법론의 상관** — 발행 전 한국 영업일 win 일(=6개월) 창에서 쌍별 Pearson.

    calendar 는 KRX 거래일 DatetimeIndex (보통 ^KS200 수익률의 인덱스).
    3시장 공통 거래일로 자르는 corr180 과 달리 **쌍별로** 두 시장이 모두 거래한 날만 쓴다
    (공통일로 자르면 창이 6.2~6.9개월로 늘어난다). 비PSD 는 엔진의 chol_psd 가 보정한다.

    입력
      rets     : list[Series] — 자산별 일별 로그수익률
      dt       : Timestamp — 발행일
      calendar : DatetimeIndex — KRX 거래일 (보통 ^KS200 수익률의 인덱스)
      win      : int — 창 길이 [한국 영업일]. 기본 120 = 6개월
    출력
      (C, start) — C : ndarray(k, k), ±0.999 로 클립하고 대각을 1 로 맞춘 상관행렬
                   start : Timestamp, 창 시작일
                   달력이 win 보다 짧거나 쌍 하나라도 60일을 못 채우면 (None, None)
    """
    bd = calendar[calendar < dt]
    if len(bd) < win:
        return None, None
    w = pd.concat([r[(r.index >= bd[-win]) & (r.index < dt)] for r in rets], axis=1, sort=True)
    C = w.corr(min_periods=60).values
    if np.isnan(C).any():
        return None, None
    C = np.clip(C, -0.999, 0.999)
    np.fill_diagonal(C, 1.0)
    return C, bd[-win]


def corr180(rets, dt, win=CORR_WIN):
    """발행 전 win 거래일 **3시장 공통 거래일**로 정렬한 Pearson 상관행렬.

    `corr_bd` 와 달리 세 자산이 모두 거래한 날만 쓴다 — 그래서 실제 창이 달력상 6.2~6.9개월로
    늘어난다. 현행 방법론은 `corr_bd` 쪽이다.

    입력
      rets : list[Series] — 자산별 일별 로그수익률
      dt   : Timestamp — 발행일
      win  : int — 창 길이 [공통 거래일]. 기본 CORR_WIN(180)
    출력
      ndarray(3, 3) | None — ±0.999 클립, 대각 1.
                             자산이 3개 미만이거나 공통일이 60 미만이면 None
    """
    dfr = pd.concat([r[r.index < dt] for r in rets], axis=1).dropna().tail(win)
    if dfr.shape[1] < 3 or len(dfr) < 60:
        return None
    C = np.corrcoef(dfr.values, rowvar=False)
    C = np.clip(C, -0.999, 0.999)
    np.fill_diagonal(C, 1.0)
    return C
