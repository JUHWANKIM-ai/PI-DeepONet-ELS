# -*- coding: utf-8 -*-
"""금리 커브 파일의 위치 규약 — 산출물이 아니라 원천이다.

 CD 91일 · IRS par 금리 · NS 커브 노드는 ECOS·본드웹·FRED 에서 받아오는 **수집물**이고,
 우리 코드가 만들지 않는다. 그런데 종전에는 `data/cache/` 에 있어 산출물과 섞여 있었다.
 `data/raw/rates/` 로 옮기고 경로 해석을 `fm.rate()` 한 곳으로 모은다.

 Underlying 폴더(share_underlyings_v2)에는 금리가 없다 — 구분이 회사 121 · 지수 34 ·
 원자재 3 · ETF 1 로 금리 항목이 0건이다. 그래서 금리만 원천에 따로 둔다.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm          # noqa: E402
from module import curve as CV                # noqa: E402
from module import market as MK               # noqa: E402

NAMES = ["krw_curve.parquet", "krw_curve_fred.parquet", "irs_bondweb.parquet"]


def test_rate_resolves_under_raw():
    assert fm.rate("krw_curve.parquet") == fm.RAW / "rates" / "krw_curve.parquet"


def test_all_three_curve_files_live_in_raw():
    missing = [n for n in NAMES if not fm.rate(n).exists()]
    assert missing == []


def test_curve_files_are_gone_from_cache():
    """캐시에 남아 있으면 어느 쪽을 읽는지 모호해진다."""
    stale = [n for n in NAMES if (fm.CACHE / n).exists()]
    assert stale == []


def test_market_inputs_reads_from_raw():
    CV._IRS = CV._CD_M = None
    mi = CV.market_inputs(pd.Timestamp("2024-06-03"))
    assert mi is not None
    assert "cd" in mi and "swaps" in mi


def test_ns_zero_reads_from_raw():
    dz = MK.ns_zero(pd.Timestamp("2024-06-03"))
    assert dz is not None
    assert float(dz([3.0])[0]) == pytest.approx(0.03, abs=0.02)
