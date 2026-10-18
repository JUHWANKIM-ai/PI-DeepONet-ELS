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
    """master.csv (159행) 를 읽어 돌려준다. 첫 호출에서만 읽고 모듈 전역에 캐시한다.

    입력
      없음
    출력
      DataFrame — UDLY_ID · 가격티커(yfinance) · ETF티커(yfinance) · 파일 · ETF_파일 열.
                  **같은 객체를 매번 돌려준다** — 호출부가 수정하면 다른 호출부가 본다
    """
    global _MASTER
    if _MASTER is None:
        _MASTER = pd.read_csv(fm.DATA / SU_DIR / "master.csv", encoding="utf-8-sig")
    return _MASTER


def _read_px(fn):
    """share_underlyings_v2 의 CSV 하나를 Close_Adj 계열로 읽는다.

    입력
      fn : str — SU_DIR 안의 파일명 (master.csv 의 `파일` 또는 `ETF_파일` 값)
    출력
      Series | None — DatetimeIndex 오름차순, 중복 날짜 제거, NaN 제거.
                      유효 관측이 하나도 없으면 None
    """
    d = pd.read_csv(fm.DATA / SU_DIR / fn)
    d["Date"] = pd.to_datetime(d.Date)
    s = d.set_index("Date")["Close_Adj"].dropna()
    return s[~s.index.duplicated()].sort_index() if len(s) else None


def price(tk, udly=None):
    """티커 -> 일별 종가 Series (Close_Adj, DatetimeIndex 오름차순). 매핑 없으면 None.

    상장 전 구간은 **비어 있다** — 0 으로 채우지 않는다. 그래야 변동성이 NaN 이 되어
    그 상품이 명시적으로 탈락한다.

    `udly`: 티커로 못 찾을 때 쓸 UDLY_ID. LSEG 가 yfinance 티커를 주지 않는 기초자산이 있고
    (^SX7E = EURO STOXX Banks 는 `가격티커(yfinance)` 가 비어 있다) master.csv 의 키는
    UDLY_ID 다. 티커로만 찾으면 그런 자산을 참조하는 상품이 조용히 사라진다 — ^SX7E 는
    DART 공시 219건이 참조하는데 데이터셋에 0건 들어왔다.
    티커로 찾히면 그 결과가 이긴다 — UDLY_ID 는 **대체 경로**지 우선 경로가 아니다.

    입력
      tk   : str — 가격 티커. 본체(`가격티커`) 를 먼저 보고 없으면 ETF 티커를 본다
      udly : str | None — 티커로 못 찾을 때 쓸 UDLY_ID
    출력
      Series | None — 일별 종가(Close_Adj), DatetimeIndex 오름차순.
                      매핑이 없으면 None → 호출부가 그 상품을 탈락시킨다
                      (캐시 키는 (tk, udly) 쌍이다 — udly 를 주고 부른 결과와
                       주지 않고 부른 결과가 별도로 캐시된다)
    """
    key = (tk, udly)
    if key in _PX:
        return _PX[key]
    M = master()
    out = None
    for tcol, fcol in [("가격티커(yfinance)", "파일"), ("ETF티커(yfinance)", "ETF_파일")]:
        r = M[M[tcol].astype(str) == tk]
        if len(r) and isinstance(r.iloc[0][fcol], str):
            out = _read_px(r.iloc[0][fcol])
            if out is not None:
                break
    if out is None and udly is not None:
        r = M[M["UDLY_ID"].astype(str) == str(udly)]
        if len(r) and isinstance(r.iloc[0]["파일"], str):
            out = _read_px(r.iloc[0]["파일"])
    _PX[key] = out
    return out


def prices(tks):
    """티커 목록 -> {티커: Series}. 매핑된 것만 담는다.

    **UDLY_ID 대체 경로를 쓰지 않는다** — `price(t)` 를 udly 없이 부르므로, 가격 티커가 빈
    기초자산(^SX7E 등)은 여기서 조용히 빠진다. 그 자산이 필요한 호출부는
    `price(tk, udly=...)` 를 직접 쓴다 (`module.mc.returns_by_ticker` 가 그렇게 한다).

    입력
      tks : 반복자[str] — 티커 목록
    출력
      dict[str -> Series] — 찾힌 것만. 못 찾은 티커는 키 자체가 없다
    """
    got = {}
    for t in tks:
        s = price(t)
        if s is not None:
            got[t] = s
    return got
