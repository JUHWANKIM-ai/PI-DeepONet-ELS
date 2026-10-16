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
         "^N225": "1321.T", "^HSI": "2800.HK", "^NDX": "QQQ"}
WINDOW_DAYS = 320                  # 최근 분배일로 끝나는 1년 창 (분기배당 4회를 담되 중복은 피하는 폭)
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


def build_div(tks, t0, ten, skip=()):
    """(divlog[N,3], qcont[3], 이벤트수, 이벤트목록).

     divlog 는 배당락일에 log(1−δ) 를 넣은 일별 드리프트 보정.
     qcont 는 같은 총량을 연속 배당수익률로 환산한 것 (비교용).
     skip: 배당을 0 으로 둘 티커 집합 (예: ^GDAXI 는 총수익지수라 배당이 이미 반영돼 있다)."""
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
