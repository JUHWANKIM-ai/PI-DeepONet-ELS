# -*- coding: utf-8 -*-
"""가격 공급원 전환 — PX_SOURCE 환경변수.

 `cache`      현행 `data/cache/px_*.parquet` (배당 역조정 = 총수익 계열)
 `underlying` `data/share_underlyings_v2` 의 Close_Adj (액면분할만 조정)

 **기본값은 `underlying` 이다.** 캐시는 입력이 아니라 산출물이므로, 캐시를 격리한 뒤에도
 아무것도 막히지 않아야 한다. `cache` 는 구 계열과의 대조(generate/measure_source_switch.py)에서만 쓴다.

 두 계열은 개별주식에서 실제로 다르다(삼성전자 2015-01-02: 20,419 vs 26,600).
 지수는 거의 같다(^KS200 수익률 상관 0.99994).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm          # noqa: E402
import module.mc as MC                        # noqa: E402
import module.underlying as UL                # noqa: E402

TK = "005930.KS"

# px_* 캐시는 원천 전용 실행에서 scratch 로 격리된다. 캐시 경로를 재는 테스트는
# 그 픽스처가 있어야만 의미가 있으므로, 없으면 실패가 아니라 건너뛴다.
_HAS_CACHE = (fm.CACHE / "px_005930_KS.parquet").exists()
needs_cache = pytest.mark.skipif(not _HAS_CACHE, reason="px_* 캐시가 없다 (원천 전용 트리)")


def _ret(monkeypatch, src):
    if src is None:
        monkeypatch.delenv("PX_SOURCE", raising=False)
    else:
        monkeypatch.setenv("PX_SOURCE", src)
    return MC.load_market()["RET"][TK].dropna()


def test_default_source_is_underlying(monkeypatch):
    """캐시를 격리한 뒤로는 기본값이 underlying 이어야 아무것도 안 막힌다."""
    a = _ret(monkeypatch, None)
    b = _ret(monkeypatch, "underlying")
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_underlying_source_matches_adapter(monkeypatch):
    got = _ret(monkeypatch, "underlying")
    want = np.log(UL.price(TK)).diff().dropna()
    pd.testing.assert_series_equal(got, want, check_names=False)


@needs_cache
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


def test_build_source_inputs_honor_px_source(monkeypatch):
    """build_source 도 같은 전환을 따라야 한다 — px_* 를 직접 읽는 두 번째 경로였다.

     load_market 만 고치면 원천 전용 실행에서 build_source 가 전 상품을 탈락시킨다."""
    import module.build_source as BS
    monkeypatch.setenv("PX_SOURCE", "underlying")
    ctx = BS.load_inputs()
    assert len(ctx.RET) > 20
    got = ctx.RET["005930.KS"].dropna()
    want = np.log(UL.price("005930.KS")).diff().dropna()
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_returns_by_ticker_is_the_single_source_of_truth(monkeypatch):
    """가격 적재는 한 함수에서만 일어나야 한다.

     같은 로직이 module.mc.load_market · module.build_source.load_inputs ·
     generate.build_12star.main 세 곳에 복사돼 있었고, 원천 전용 실행에서 한 곳씩 차례로 터졌다."""
    monkeypatch.setenv("PX_SOURCE", "underlying")
    mapping = {"A": "^KS200", "B": "005930.KS", "C": None, "D": "__NO_SUCH__"}
    got = MC.returns_by_ticker(mapping)
    assert set(got) == {"^KS200", "005930.KS"}
    pd.testing.assert_series_equal(
        got["005930.KS"], np.log(UL.price("005930.KS")).diff(), check_names=False)


def test_build_12star_uses_the_shared_loader():
    import generate.build_12star as B12
    import inspect
    src = inspect.getsource(B12.main)
    assert "returns_by_ticker" in src, "build_12star 가 가격 적재를 또 복사했다"
    assert "_safe(t)" not in src, "build_12star 에 px_* 직접 읽기가 남아 있다"
