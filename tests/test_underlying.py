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
import json
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


# 가격이 없는 기초자산은 **데이터셋에 들어올 수 없다** — build_source 가 "가격이력 없음" 으로
# 그 상품을 떨어뜨린다. 그래서 데이터셋을 훑는 커버리지 테스트는 동어반복이고 절대 실패하지 않는다.
# 검사 대상은 **유니버스**(udly_ticker_map.json)여야 한다.
#
# 아래는 지금 해결되지 않는 것들이다. 알려진 공백으로 명시해 두고, **새** 공백만 빨간불이 되게 한다.
#   ^SX7E  EURO STOXX BANKS — master.csv 에 행은 있으나(EU0009658426, 가격 6,863행 + ETF EXX1.DE)
#          `가격티커(yfinance)` 가 비어 있어 티커로는 찾을 수 없다. DART 공시 219건이 참조하고
#          데이터셋에는 0건 들어온다. 어댑터가 UDLY_ID 로도 찾게 하면 복구되지만 유니버스가
#          바뀌므로(지수 1종·상품 219건 추가) 사람 판단이 필요하다.
KNOWN_UNMAPPED = {"^SX7E"}


def test_every_mapped_ticker_in_the_universe_resolves():
    """유니버스의 티커가 전부 가격으로 해석돼야 한다 — 안 되면 상품이 조용히 사라진다.

     데이터셋이 아니라 **매핑 파일**을 훑는다. 데이터셋을 훑으면 이미 탈락한 것을 볼 수 없다."""
    m = json.loads((fm.RAW / "udly_ticker_map.json").read_text(encoding="utf-8"))
    tks = sorted({v for v in m.values() if v})
    bad = {t for t in tks if UL.price(t) is None}
    assert bad == KNOWN_UNMAPPED, "새로 해석되지 않는 티커: {}".format(sorted(bad - KNOWN_UNMAPPED))


def test_known_unmapped_tickers_cost_is_recorded():
    """알려진 공백이 실제로 상품을 떨어뜨리고 있다는 사실을 수치로 못박는다.

     공백이 저절로 메워졌다면(그 경우 이 테스트가 빨간불) KNOWN_UNMAPPED 에서 지워야 한다."""
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet", columns=["udl1", "udl2", "udl3"])
    for t in KNOWN_UNMAPPED:
        assert int((d == t).any(axis=1).sum()) == 0, "{} 가 데이터셋에 들어왔다".format(t)
        assert UL.price(t) is None, "{} 가 이제 해석된다 — KNOWN_UNMAPPED 에서 지워라".format(t)


def test_late_listed_ticker_has_no_prices_before_listing():
    """상장 전 구간은 빈 구간이어야 한다 — 0 으로 채우면 변동성이 조용히 틀린다.

     373220.KS(LG에너지솔루션)는 2022-01-27 상장."""
    s = UL.price("373220.KS")
    assert s.index.min() >= pd.Timestamp("2022-01-27")
    assert len(s[s.index < pd.Timestamp("2022-01-01")]) == 0


def test_prices_returns_only_mapped():
    got = UL.prices(["^KS200", "__NO_SUCH_TICKER__", "005930.KS"])
    assert set(got) == {"^KS200", "005930.KS"}
