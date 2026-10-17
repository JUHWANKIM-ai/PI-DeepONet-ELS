# -*- coding: utf-8 -*-
"""배당 소스 분기는 한 함수에서만 일어나야 한다.

 `if DIV == "lseg": qcont ... else: build_div ...` 가 tools/build_mc_variants.py 에 두 번,
 그리고 tools/mc_check_12star.py 는 그 분기 없이 `DV.build_div` 를 직접 불렀다.
 그래서 원천 전용 실행에서 노트북 1 의 세 번째 단계가 `data/cache/div_SPY.parquet` 을 찾다 죽었다
 — 가격 적재가 세 곳에 복사돼 있던 것과 같은 결함이다.

 `module.dividend.div_kwargs` 가 유일한 분기점이고, 세 호출부가 전부 그것을 부른다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm          # noqa: E402
from module import dividend as DV             # noqa: E402

TKS = ["^KS200", "005930.KS", "^GSPC"]
T0 = pd.Timestamp("2022-06-01")
TEN = 3.0

_HAS_DIVCACHE = (fm.CACHE / "div_SPY.parquet").exists()
needs_divcache = pytest.mark.skipif(not _HAS_DIVCACHE, reason="div_* 캐시가 없다 (원천 전용 트리)")


def test_lseg_gives_continuous_q_only():
    """LSEG 에는 배당락일이 없다 — 엔진에 qcont 만 넘겨야 한다."""
    dkw, qc, nev = DV.div_kwargs(TKS, T0, TEN, "lseg")
    assert set(dkw) == {"qcont"}
    assert nev == 0
    assert np.isfinite(qc).all() and len(qc) == 3


def test_lseg_path_never_reads_div_cache(monkeypatch):
    """원천 전용 트리에서 div_* 를 읽으면 FileNotFoundError 로 죽는다."""
    def boom(tk):
        raise AssertionError("lseg 경로가 div_* 캐시를 읽었다: {}".format(tk))
    monkeypatch.setattr(DV, "div_hist", boom)
    dkw, _, _ = DV.div_kwargs(TKS, T0, TEN, "lseg")
    assert "qcont" in dkw


@needs_divcache
def test_yf_gives_discrete_divlog():
    """정본 비교용 이산배당 경로는 남아 있어야 한다."""
    dkw, qc, nev = DV.div_kwargs(TKS, T0, TEN, "yf")
    assert set(dkw) == {"divlog"}
    assert dkw["divlog"].shape == (int(round(TEN * 365)), 3)
    assert nev > 0


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="배당 소스"):
        DV.div_kwargs(TKS, T0, TEN, "lesg")


def test_missing_lseg_yield_returns_none():
    """수익률이 없으면 호출부가 그 행을 'lseg div' 로 탈락시킬 수 있어야 한다."""
    assert DV.div_kwargs(["__NO_SUCH__", "^KS200", "^KS200"], T0, TEN, "lseg") is None


def test_every_caller_uses_the_shared_branch():
    """분기가 또 복사되면 원천 전용 실행이 같은 자리에서 다시 터진다."""
    import inspect
    import tools.build_mc_variants as BMV
    import tools.mc_check_12star as MC12
    for mod in (BMV, MC12):
        src = inspect.getsource(mod)
        assert "div_kwargs" in src, "{} 가 공용 분기를 쓰지 않는다".format(mod.__name__)
        assert "DV.build_div(" not in src, "{} 에 배당 분기가 복사돼 있다".format(mod.__name__)


def test_mc_check_12star_honors_the_default_source():
    """섭동 격자도 같은 기본값(lseg)을 써야 라벨과 참값의 배당 가정이 일치한다."""
    import tools.mc_check_12star as MC12
    assert MC12.DIV == DV.DEFAULT_SRC == "lseg"
