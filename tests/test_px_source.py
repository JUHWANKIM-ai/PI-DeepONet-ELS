# -*- coding: utf-8 -*-
"""가격 공급원 전환 — PX_SOURCE 환경변수.

 `cache`      현행 `data/cache/px_*.parquet` (배당 역조정 = 총수익 계열)
 `underlying` `data/share_underlyings_v2` 의 Close_Adj (액면분할만 조정)

 기본값은 `cache` 로 둔다 — 그래야 저장된 라벨의 비트 단위 재현이 유지된다.
 전환은 원천 전용 재현(Task 6)에서만 켠다.

 두 계열은 개별주식에서 실제로 다르다(삼성전자 2015-01-02: 20,419 vs 26,600).
 지수는 거의 같다(^KS200 수익률 상관 0.99994).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import module.mc as MC                        # noqa: E402
import module.underlying as UL                # noqa: E402

TK = "005930.KS"


def _ret(monkeypatch, src):
    if src is None:
        monkeypatch.delenv("PX_SOURCE", raising=False)
    else:
        monkeypatch.setenv("PX_SOURCE", src)
    return MC.load_market()["RET"][TK].dropna()


def test_default_source_is_cache(monkeypatch):
    a = _ret(monkeypatch, None)
    b = _ret(monkeypatch, "cache")
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_underlying_source_matches_adapter(monkeypatch):
    got = _ret(monkeypatch, "underlying")
    want = np.log(UL.price(TK)).diff().dropna()
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_two_sources_differ_for_a_company(monkeypatch):
    """배당 역조정 차이가 수익률에 실제로 나타난다."""
    a = _ret(monkeypatch, "cache")
    b = _ret(monkeypatch, "underlying")
    j = pd.concat([a.rename("c"), b.rename("u")], axis=1).dropna()
    assert len(j) > 1000
    assert (j.c - j.u).abs().max() > 1e-4


def test_unknown_source_is_rejected(monkeypatch):
    """오타가 조용히 캐시로 떨어지면 안 된다."""
    monkeypatch.setenv("PX_SOURCE", "underlyng")
    with pytest.raises(ValueError, match="PX_SOURCE"):
        MC.load_market()
