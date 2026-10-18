# -*- coding: utf-8 -*-
"""원천 전용 재현의 배당·경로 수 기본값.

**배당.** Underlying 폴더에는 배당락일·금액이 없고 일별 배당수익률 지표만 있다
(`Dividend_Yield_pct`). 그래서 이산배당을 만들 수 없고 연속 배당수익률 q 로만 쓸 수 있다.
원천 전용으로 가려면 기본값이 `lseg` 여야 한다.

**경로 수.** 방법론 목표는 100,000 이다. 측정상 40,000 대비 잡음이 1.71배 줄고
(평균|차이| 6.20 → 3.62원, 40만 경로를 준참값으로) 편향은 +0.23 → +0.29원으로 변화가 없다.

지수는 LSEG 가 배당수익률을 주지 않으므로 추종 ETF 로 대신 조회한다 — 그 매핑이
데이터셋 전 기초자산에 대해 성립하는지를 여기서 못박는다 (Review Focus 5).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm  # noqa: E402
from module import dividend as DV  # noqa: E402
import module.market as MK  # noqa: E402

# 이산배당(yf) 분기는 `data/cache/div_*.parquet` 이 있어야만 돌 수 있다. 원천 전용 트리에서는
# 격리돼 있으므로 실패가 아니라 건너뛴다.
needs_divcache = pytest.mark.skipif(
    not (fm.CACHE / "div_SPY.parquet").exists(), reason="div_* 캐시가 없다 (원천 전용 트리)"
)


@pytest.fixture(scope="module")
def mkt():
    ret = MK.returns()
    return ret, MK.calendar(ret)


@pytest.fixture(scope="module")
def row():
    return MK.dataset().iloc[0]


def test_every_dataset_ticker_has_lseg_yield():
    """ETF 프록시까지 포함해 전 기초자산에 배당수익률이 있어야 한다 — 없으면 전건 탈락한다."""
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet")
    tks = (set(d.udl1) | set(d.udl2) | set(d.udl3)) - set(DV.ZERO_DIV)
    missing = [
        t
        for t in sorted(tks)
        if DV.lseg_yield(DV.PROXY.get(t, t)) is None and DV.lseg_yield(t) is None
    ]
    assert missing == []


def test_state_for_defaults_to_continuous_q(row, mkt):
    st = MK.state_for(row, *mkt)
    assert st["div"] == "lseg"
    assert st["divlog"] is None
    assert st["ndiv"] == 0
    assert np.isfinite(st["qcont"]).all()


@needs_divcache
def test_discrete_dividend_is_still_reachable(row, mkt):
    """정본 비교용으로 이산배당 경로는 남아 있어야 한다."""
    st = MK.state_for(row, *mkt, div="yf")
    assert st["div"] == "yf" and st["divlog"] is not None


def test_npath_is_methodology_target():
    import generate.build_mc_variants as BMV

    assert BMV.NPATH == 100_000


def test_default_label_file_has_no_suffix():
    """기본값이 접미사 없는 이름이어야 노트북·module.mcvar 가 그대로 읽는다."""
    import generate.build_mc_variants as BMV

    assert BMV.OUT_NAME == "mc_variants.parquet"
    assert BMV.OUT12_NAME == "mc_variants12.parquet"
