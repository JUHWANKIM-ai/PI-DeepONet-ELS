# -*- coding: utf-8 -*-
"""share_underlyings_v2 가격 어댑터 — 원천 전용 재현의 가격 공급원.

 `data/share_underlyings_v2/` 는 기초자산 159종의 일별 계열을 담는다.
 각 CSV 의 열은 Date · Close_Adj · Close_Raw · Dividend_Yield_pct 넷이다.

   Close_Adj   액면분할만 조정한 주가   <- 이 모듈이 쓰는 것
   Close_Raw   조정 없음 (분할 시점에 튄다)
   Dividend_Yield_pct  배당수익률 지표 (module/dividend.lseg_yield 가 읽는다)

 **왜 Close_Adj 인가.** MC 엔진은 배당을 divlog(이산) 또는 qcont(연속)로 드리프트에서 따로 뺀다.
 그러므로 변동성·상관은 배당이 역조정되지 않은 주가 수익률로 재야 이중계산이 없다.
 현행 `data/cache/px_*.parquet` 은 배당까지 역조정된 총수익 계열이라 이 점에서 어긋난다
 (삼성전자 2015-01-02: px_ 캐시 20,419 vs Close_Adj 26,600).

 매핑은 master.csv 의 기초자산 본체를 먼저 보고, 없으면 관련 ETF 를 본다 —
 module/dividend.lseg_yield 와 같은 순서다.
"""
import pandas as pd

from util import file_manager as fm

SU_DIR = "share_underlyings_v2"
_MASTER = None
_PX = {}


def master():
    """master.csv (159행). 캐시된다."""
    global _MASTER
    if _MASTER is None:
        _MASTER = pd.read_csv(fm.DATA / SU_DIR / "master.csv", encoding="utf-8-sig")
    return _MASTER


def price(tk):
    """티커 -> 일별 종가 Series (Close_Adj, DatetimeIndex 오름차순). 매핑 없으면 None.

     상장 전 구간은 **비어 있다** — 0 으로 채우지 않는다. 그래야 변동성이 NaN 이 되어
     그 상품이 명시적으로 탈락한다."""
    if tk in _PX:
        return _PX[tk]
    M = master()
    out = None
    for tcol, fcol in [("가격티커(yfinance)", "파일"), ("ETF티커(yfinance)", "ETF_파일")]:
        r = M[M[tcol].astype(str) == tk]
        if len(r) and isinstance(r.iloc[0][fcol], str):
            d = pd.read_csv(fm.DATA / SU_DIR / r.iloc[0][fcol])
            d["Date"] = pd.to_datetime(d.Date)
            s = d.set_index("Date")["Close_Adj"].dropna()
            if len(s):
                out = s[~s.index.duplicated()].sort_index()
                break
    _PX[tk] = out
    return out


def prices(tks):
    """티커 목록 -> {티커: Series}. 매핑된 것만 담는다."""
    got = {}
    for t in tks:
        s = price(t)
        if s is not None:
            got[t] = s
    return got
