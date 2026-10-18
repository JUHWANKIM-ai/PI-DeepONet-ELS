# -*- coding: utf-8 -*-
"""state_for 의 특성화 테스트 — 현행 방법론의 '발행일 시장상태' 조립 규약을 못박는다.

모든 MC 가격이 이 함수를 통과하는데(module/mc.py · generate/build_mc_variants · generate/build_decomp ·
generate/mc_check_12star) 줄 커버리지가 5% 였다 (CRAP 135.5).

여기서 고정하는 것:
  1. 성공 상태의 키 집합과 모양 — sig(3) · corr(3x3) · dz(콜러블) · divlog(N x 3)
  2. 네 가지 탈락 — 수익률 결측 / 상관 결측 / 커브 결측 / 변동성 NaN
  3. 배당 소스 분기 — yf 는 이산 divlog(+이벤트수), LSEG 는 연속 qcont(divlog None, ndiv 0)
  4. dt 는 isu_ord 에서 오고, divlog 창 길이는 그 상품의 tenor 에서 온다

인과성 자체(발행일 이전 자료만 보는지)는 module/features.py 쪽 관심사라 여기서 다루지 않는다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module import curve as CV  # noqa: E402
from module import dividend as DV  # noqa: E402
from module import features as F  # noqa: E402
import module.market as MK  # noqa: E402
from util import file_manager as fm  # noqa: E402

# dkw = mc_engine 에 그대로 풀어 넘길 배당 kwargs. divlog 를 직접 꺼내는 호출부가
# lseg(기본값)에서 배당을 조용히 잃어서 추가했다 — tests/test_div_source_switch.py 참고.
KEYS = {"dt", "tks", "sig", "corr", "dz", "dkw", "divlog", "qcont", "ndiv", "div"}

# yf 이산배당 분기는 `data/cache/div_*.parquet` 이 있어야만 돌 수 있다. 원천 전용 트리에서는
# 그 캐시가 격리돼 있으므로 실패가 아니라 건너뛴다 — 못 도는 테스트를 빨간불로 두면
# 진짜 실패가 묻힌다.
needs_divcache = pytest.mark.skipif(
    not (fm.CACHE / "div_SPY.parquet").exists(), reason="div_* 캐시가 없다 (원천 전용 트리)"
)


@pytest.fixture(scope="module")
def mkt():
    ret = MK.returns()
    return ret, MK.calendar(ret)


@pytest.fixture(scope="module")
def row():
    """정본 데이터셋의 첫 상품 — 시장상태가 실제로 조립되는 행."""
    return MK.dataset().iloc[0]


def test_returns_full_state(row, mkt):
    """성공 상태의 키 집합과 모양."""
    st = MK.state_for(row, *mkt)
    assert st is not None
    assert set(st) == KEYS
    assert len(st["sig"]) == 3 and np.isfinite(st["sig"]).all()
    assert np.asarray(st["corr"]).shape == (3, 3)
    assert callable(st["dz"])
    assert st["div"] == "lseg"
    assert st["tks"] == [row["udl1"], row["udl2"], row["udl3"]]


def test_dt_comes_from_isu_ord(row, mkt):
    """발행일은 isu_ord(서수)에서 복원한다 — 날짜 문자열을 다시 파싱하지 않는다."""
    st = MK.state_for(row, *mkt)
    assert st["dt"] == pd.Timestamp.fromordinal(int(row["isu_ord"]))


def test_corr_is_symmetric_with_unit_diagonal(row, mkt):
    """상관행렬 규약 — 대각 1, 대칭."""
    C = np.asarray(MK.state_for(row, *mkt)["corr"], float)
    assert np.allclose(np.diag(C), 1.0)
    assert np.allclose(C, C.T)


@needs_divcache
def test_divlog_window_follows_tenor(row, mkt):
    """divlog 창 길이는 그 상품의 tenor 에서 온다 (build_div 의 N = round(ten*365)).

    기본값이 연속 q 로 바뀌었으므로 이산배당 분기를 명시해서 부른다."""
    st = MK.state_for(row, *mkt, div="yf")
    assert st["divlog"].shape == (int(round(float(row["tenor"]) * 365)), 3)


@needs_divcache
def test_yf_branch_passes_tenor_to_build_div(monkeypatch, row, mkt):
    """yf 분기는 그 상품의 tenor 로 이산배당을 만든다 — 고정 만기를 쓰지 않는다."""
    # 기본값이 lseg 로 바뀌었으므로 이 분기는 명시해서 부른다
    seen = {}
    real = DV.build_div

    def spy(tks, t0, ten, **kw):
        seen.update(tks=tks, t0=t0, ten=ten)
        return real(tks, t0, ten, **kw)

    monkeypatch.setattr(DV, "build_div", spy)
    st = MK.state_for(row, *mkt, div="yf")
    assert seen["ten"] == pytest.approx(float(row["tenor"]))
    assert seen["t0"] == st["dt"] and seen["tks"] == st["tks"]


def test_none_when_underlying_return_missing(row, mkt):
    """기초자산 수익률이 하나라도 없으면 None (가격 캐시에 없는 티커)."""
    bad = row.copy()
    bad["udl2"] = "__NO_SUCH_TICKER__"
    assert MK.state_for(bad, *mkt) is None


def test_none_when_corr_missing(monkeypatch, row, mkt):
    """상관 결측(공통 영업일 부족) -> None."""
    monkeypatch.setattr(F, "corr_bd", lambda *a, **k: (None, None))
    assert MK.state_for(row, *mkt) is None


def test_none_when_curve_missing(monkeypatch, row, mkt):
    """CD/IRS 커브 결측 -> None."""
    monkeypatch.setattr(CV, "zero_from_date", lambda *a, **k: None)
    assert MK.state_for(row, *mkt) is None


def test_none_when_vol_is_nan(monkeypatch, row, mkt):
    """변동성이 NaN 이면 None — 가격 이력이 짧은 신규 종목."""
    monkeypatch.setattr(F, "vol_ewma_lam", lambda *a, **k: float("nan"))
    assert MK.state_for(row, *mkt) is None


def test_lseg_branch_uses_continuous_q(monkeypatch, row, mkt):
    """LSEG 분기는 연속 q 만 쓴다 — divlog 는 None, 이산 이벤트 수는 0.

    LSEG 에는 배당락일·금액이 없어 이산배당을 만들 수 없다."""
    q = np.array([0.011, 0.022, 0.033], "float32")
    monkeypatch.setattr(DV, "qcont_lseg", lambda *a, **k: q)
    monkeypatch.setattr(
        DV, "build_div", lambda *a, **k: pytest.fail("LSEG 에서 이산배당을 부르면 안 된다")
    )
    st = MK.state_for(row, *mkt, div="lseg")
    assert st is not None
    assert st["divlog"] is None and st["ndiv"] == 0
    assert st["div"] == "lseg"
    assert np.allclose(st["qcont"], q)


def test_lseg_none_when_yield_missing(monkeypatch, row, mkt):
    """LSEG 수익률이 없는 기초자산이면 None — yf 분기와 탈락 조건이 다르다."""
    monkeypatch.setattr(DV, "qcont_lseg", lambda *a, **k: None)
    assert MK.state_for(row, *mkt, div="lseg") is None


def test_lseg_is_the_default_source(row, mkt):
    """div 를 안 주면 lseg(연속 q) 다 — 원천 전용 재현 경로.

    Underlying 폴더에 배당락일·금액이 없어 이산배당을 만들 수 없기 때문이다."""
    a = MK.state_for(row, *mkt)
    b = MK.state_for(row, *mkt, div="lseg")
    assert a["div"] == b["div"] == "lseg"
    assert a["divlog"] is None
    assert np.allclose(a["qcont"], b["qcont"])


def test_market_data_loaded_lazily_when_not_passed(row):
    """ret·cal 을 안 넘기면 스스로 적재한다 (호출자 편의 경로)."""
    st = MK.state_for(row)
    assert st is not None and set(st) == KEYS
