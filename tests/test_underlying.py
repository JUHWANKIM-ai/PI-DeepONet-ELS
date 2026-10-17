# -*- coding: utf-8 -*-
"""share_underlyings_v2 가격 어댑터 — 원천 전용 재현의 가격 공급원.

 현행은 `data/cache/px_*.parquet`(배당 역조정 = 총수익 계열)을 읽는다.
 Underlying 폴더의 `Close_Adj` 는 **액면분할만** 조정한 주가다 — 둘은 다르고,
 MC 엔진이 배당을 divlog/qcont 로 따로 빼므로 방법론상 후자가 맞다.

 여기서 못박는 것:
   1. 지수와 개별주식 모두 매핑된다 (본체 파일 -> ETF 파일 순서)
   2. Close_Adj 는 분할조정·배당미조정이다 (삼성전자 2018년 50:1 분할 기준점)
   3. 매핑 없는 티커는 None (조용히 빈 Series 를 돌려주지 않는다)
   4. 데이터셋 기초자산 44종이 모두 매핑된다
   5. 상장 전 구간은 비어 있다 (0 으로 채우지 않는다)
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm          # noqa: E402
import module.underlying as UL                # noqa: E402


def test_price_maps_index_and_company():
    assert UL.price("^KS200") is not None
    assert UL.price("005930.KS") is not None


def test_price_is_split_adjusted_not_dividend_adjusted():
    """삼성전자 2015-01-02 = 26,600원.

     분할조정만 했으면 26,600 (분할 전 1,330,000 / 50).
     배당까지 역조정한 현행 px_ 캐시는 같은 날 20,419 로 더 낮다."""
    s = UL.price("005930.KS")
    assert s.loc[pd.Timestamp("2015-01-02")] == pytest.approx(26600.0, rel=1e-6)


def test_price_returns_none_for_unknown():
    assert UL.price("__NO_SUCH_TICKER__") is None


def test_all_dataset_tickers_are_mapped():
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet")
    tks = set(d.udl1) | set(d.udl2) | set(d.udl3)
    assert [t for t in sorted(tks) if UL.price(t) is None] == []


def test_late_listed_ticker_has_no_prices_before_listing():
    """상장 전 구간은 빈 구간이어야 한다 — 0 으로 채우면 변동성이 조용히 틀린다.

     373220.KS(LG에너지솔루션)는 2022-01-27 상장."""
    s = UL.price("373220.KS")
    assert s.index.min() >= pd.Timestamp("2022-01-27")
    assert len(s[s.index < pd.Timestamp("2022-01-01")]) == 0


def test_prices_returns_only_mapped():
    got = UL.prices(["^KS200", "__NO_SUCH_TICKER__", "005930.KS"])
    assert set(got) == {"^KS200", "005930.KS"}
