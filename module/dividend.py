# -*- coding: utf-8 -*-
"""이산배당 스케줄 — 현행 방법론의 배당 입력.

 지수는 배당 이력이 제공되지 않으므로 추종 ETF 를 프록시로 쓴다(PROXY).
 발행일 t0 **이전** 최근 분배일로 끝나는 1년 창의 배당락 이벤트를 뽑아, 만기까지 매년 같은 기념일에
 되풀이된다고 보고 (N일 × 3자산) 로그하락 행렬을 만든다. mc_engine.mc_daily_t(divlog=...) 에 그대로 넘긴다.

 인과성: div_events 는 t0 이전 자료만 본다. 미래 배당 공시를 쓰지 않는다.
 종전에는 scratch/mc_discrete_div.py 에만 있어 추적 파이프라인에서 쓸 수 없었다.

 연속 근사도 같이 돌려준다(qc) — 실측상 이산과 연속의 가격 차이는 0.4원 수준이라 거의 같다.
"""
import numpy as np
import pandas as pd

from util import file_manager as fm

PROXY = {"^STOXX50E": "FEZ", "^GSPC": "SPY", "^HSCE": "FXI", "^KS200": "069500.KS",
         "^N225": "1321.T", "^HSI": "2800.HK", "^NDX": "QQQ",
         # CSI 300 은 LSEG 가 지수 자체의 배당수익률을 주지 않는다(배당수익률_행수 0).
         # 추종 ETF 510300.SS(Huatai-PB CSI 300 ETF, 벤치마크 CSI 300 TR)로 조회한다 —
         # 다른 지수와 같은 규약이다. 이 항목이 없어 CSI 300 상품 123건이 전부
         # "lseg div" 로 탈락했다. ETF 수익률은 2014-01-21 부터 있고, 해당 상품의
         # 발행일은 2019-03-08 ~ 2020-01-23 로 모두 범위 안이다.
         "000300.SS": "510300.SS"}
WINDOW_DAYS = 320                  # 최근 분배일로 끝나는 1년 창 (분기배당 4회를 담되 중복은 피하는 폭)
ZERO_DIV = frozenset({"^GDAXI"})   # 총수익지수 — 배당이 이미 지수에 반영돼 있어 배당 이력이 없다
_HIST = {}


def _safe(t):
    return t.replace(".", "_").replace("^", "_")


def div_hist(tk):
    """티커(또는 프록시 ETF)의 배당·종가 이력. 캐시 파일을 지연 적재한다."""
    s = PROXY.get(tk, tk)
    if s not in _HIST:
        _HIST[s] = pd.read_parquet(fm.CACHE / "div_{}.parquet".format(_safe(s)))
    return _HIST[s]


def div_events(tk, t0):
    """발행일 t0 이전 최근 분배일 d_k 로 끝나는 창 (d_k−320일, d_k] 의 (배당락일, δ=배당/전일종가)."""
    h = div_hist(tk)
    close, dv = h["Close"], h["Dividends"]
    ev = dv[(dv > 0) & (dv.index < t0)]
    if not len(ev):
        return []
    dk = ev.index.max()
    out = []
    for dj, Dj in ev[ev.index > dk - pd.Timedelta(days=WINDOW_DAYS)].items():
        prev = close[close.index < dj]
        if len(prev):
            out.append((dj, float(Dj / prev.iloc[-1])))
    return out


def build_div(tks, t0, ten, skip=ZERO_DIV):
    """(divlog[N,3], qcont[3], 이벤트수, 이벤트목록).

     divlog 는 배당락일에 log(1−δ) 를 넣은 일별 드리프트 보정.
     qcont 는 같은 총량을 연속 배당수익률로 환산한 것 (비교용).
     skip: 배당을 0 으로 둘 티커 집합. 기본값 ZERO_DIV 는 총수익지수를 거른다 —
           배당 이력 파일 자체가 없으므로 넘기지 않으면 FileNotFoundError 가 난다."""
    N = int(round(ten * 365))
    divlog = np.zeros((N, 3), np.float32)
    qc = np.zeros(3, np.float32)
    nev, evl = 0, []
    for i, tk in enumerate(tks):
        evs = [] if tk in skip else div_events(tk, t0)
        qc[i] = -sum(np.log1p(-y) for _, y in evs)            # 연간 총 로그하락
        for dj, y in evs:
            nd = dj
            while nd <= t0:                                    # t0 이후 첫 기념일부터
                nd = nd + pd.DateOffset(years=1)
            while (nd - t0).days <= N:
                day = (nd - t0).days
                divlog[day - 1, i] += np.log1p(-y)
                nev += 1
                evl.append((i, tk, day, nd.month, float(np.log1p(-y))))
                nd = nd + pd.DateOffset(years=1)
    return divlog, qc, nev, evl


# ---------------- LSEG 배당수익률 경로 (share_underlyings_v2) ----------------
# LSEG 는 배당 **이벤트**(배당락일 + 금액)를 주지 않고 일별 배당수익률 지표만 준다.
# 그래서 이산배당을 만들 수 없고 **연속 배당수익률 q** 로만 쓸 수 있다 (mc_engine 의 qcont).
# 실측: 이산 -> 연속 전환의 가격 영향은 평균 |차이| 7.9원으로 작고,
#       시장 공정가 대비 MAE 는 yf 이산 523.5원 / LSEG 연속 530.8원 으로 사실상 동률이다.
# 지수는 LSEG 가 배당수익률을 주지 않으므로 관련 ETF 의 값을 쓴다 (기초자산 본체 -> ETF 순서).
SU_DIR = "share_underlyings_v2"
_LS_MASTER = None
_LS_Y = {}


def _lseg_master():
    global _LS_MASTER
    if _LS_MASTER is None:
        _LS_MASTER = pd.read_csv(fm.DATA / SU_DIR / "master.csv", encoding="utf-8-sig")
    return _LS_MASTER


def lseg_yield(tk):
    """티커 -> LSEG 일별 배당수익률(소수). 본체에 없으면 관련 ETF. 없으면 None."""
    if tk in _LS_Y:
        return _LS_Y[tk]
    M = _lseg_master()
    out = None
    for tcol, fcol in [("가격티커(yfinance)", "파일"), ("ETF티커(yfinance)", "ETF_파일")]:
        r = M[M[tcol].astype(str) == tk]
        if len(r) and isinstance(r.iloc[0][fcol], str):
            d = pd.read_csv(fm.DATA / SU_DIR / r.iloc[0][fcol])
            d["Date"] = pd.to_datetime(d.Date)
            s = d.set_index("Date")["Dividend_Yield_pct"].dropna() / 100.0
            if len(s):
                out = s[~s.index.duplicated()].sort_index()
                break
    _LS_Y[tk] = out
    return out


def qcont_lseg(tks, t0, skip=ZERO_DIV):
    """발행일 t0 이전 최신 LSEG 배당수익률 3개 (연속 q). 하나라도 없으면 None.

     인과성: t0 **이전** 값만 본다. 프록시 규칙은 yfinance 쪽(PROXY)과 같은 매핑을 쓴다."""
    q = []
    for t in tks:
        if t in skip:
            q.append(0.0)
            continue
        s = lseg_yield(PROXY.get(t, t))
        if s is None:
            s = lseg_yield(t)
        if s is None:
            return None
        prev = s[s.index < t0]
        if not len(prev):
            return None
        q.append(float(prev.iloc[-1]))
    return np.asarray(q, np.float32)


# ---------------- 소스 분기 (유일한 지점) ----------------
DEFAULT_SRC = "lseg"               # 원천 전용 재현의 기본값 — LSEG 는 연속 q 로만 쓸 수 있다
SRCS = ("lseg", "yf")


def div_kwargs(tks, t0, ten, src=DEFAULT_SRC):
    """(엔진 kwargs, qcont(3), 이산 이벤트수). LSEG 수익률이 없으면 None.

     mc_engine.mc_daily_t 에 그대로 풀어 넘길 dict 를 돌려준다 —
       "lseg"  {"qcont": q}   연속 배당수익률 (이산 이벤트 없음 -> nev=0)
       "yf"    {"divlog": D}  이산 배당락 드리프트 보정

     종전에는 이 분기가 tools/build_mc_variants.py 에 두 벌 있었고 tools/mc_check_12star.py 는
     분기 없이 build_div 를 직접 불러, 원천 전용 트리에서 div_* 캐시를 찾다 죽었다."""
    if src not in SRCS:
        raise ValueError("배당 소스는 {} 중 하나여야 한다: {!r}".format(SRCS, src))
    if src == "lseg":
        qc = qcont_lseg(tks, t0)
        if qc is None:
            return None
        return dict(qcont=qc), qc, 0
    divlog, qc, nev, _ = build_div(tks, t0, float(ten))
    return dict(divlog=divlog), qc, nev
