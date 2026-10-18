# -*- coding: utf-8 -*-
"""price_row / price_row12 의 특성화(characterization) 테스트 — 지금 동작을 그대로 못박는다.

 이 두 함수는 라벨 파일(mc_variants·mc_variants12)의 **열 규약**을 정하는 곳이고,
 module/mcvar.py 와 module/star.py 가 그 규약에 맞춰 배열을 조립한다.
 그런데 커버리지가 0% 였다 (CRAP 240 / 210).

 여기서 고정하는 것은 세 가지다.
   1. 성공 레코드의 열 집합 (라벨 파일 스키마)
   2. 탈락 분기 — 시장입력 결측("inputs") / LSEG 배당 결측("lseg div")
   3. 배당 소스 분기 — yf 는 이산 divlog, LSEG 는 연속 qcont 를 엔진에 넘긴다

 MC 엔진 자체는 tests/test_mc_engine.py 가 덮으므로 여기서는 mc_daily_t 를 상수로 바꿔
 **오케스트레이션만** 본다 (그래서 즉시 끝난다).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                # noqa: E402
from module import curve as CV                     # noqa: E402
from module import dividend as DV                  # noqa: E402
from module import features as F                   # noqa: E402
from module import market as MK                    # noqa: E402
from module import mc_engine as ME                 # noqa: E402
from generate import build_mc_variants as BMV         # noqa: E402

SA = ["sA_1", "sA_2", "sA_3"]
SB = ["sB_1", "sB_2", "sB_3"]
RHO = ["rho_12", "rho_13", "rho_23"]
QQ = ["q_1", "q_2", "q_3"]
BU = ["bu{}".format(j) for j in range(10)]


@pytest.fixture
def row():
    """정본 데이터셋의 첫 상품 — 시장입력이 실제로 조립되는 행."""
    return MK.dataset().iloc[0]


@pytest.fixture
def row12():
    return BMV.dataset12().iloc[0]


@pytest.fixture(autouse=True)
def stub_engine(monkeypatch):
    """엔진은 상수로. 호출 인자를 기록해 배당 분기를 확인한다."""
    calls = []

    def fake(*a, **kw):
        calls.append(kw)
        return 0.95

    monkeypatch.setattr(ME, "mc_daily_t", fake)
    return calls


def test_price_row_record_schema(row):
    """성공 레코드의 열 집합 = 라벨 파일 스키마. 바뀌면 mcvar.vc_block 이 깨진다."""
    rec = BMV.price_row(0, row)
    assert "fail" not in rec, rec.get("fail")
    assert set(rec) == set(["i", "item", "mc_A", "mc_B", "n_div", "r_bt", "r_ns"]
                           + SA + SB + RHO + QQ + BU)
    assert rec["i"] == 0 and rec["item"] == row["item"]
    assert np.isfinite([rec[k] for k in SA + SB + RHO + QQ + BU]).all()


def test_price_row_fails_on_missing_corr(monkeypatch, row):
    """상관 결측 -> fail='inputs', 레코드는 i·item·fail 만."""
    monkeypatch.setattr(F, "corr_bd", lambda *a, **k: (None, None))
    rec = BMV.price_row(7, row)
    assert rec == {"i": 7, "item": row["item"], "fail": "inputs"}


def test_price_row_fails_on_missing_curve(monkeypatch, row):
    """CD/IRS 원자료 결측도 같은 탈락으로 묶인다."""
    monkeypatch.setattr(CV, "market_inputs", lambda *a, **k: None)
    assert BMV.price_row(0, row)["fail"] == "inputs"


needs_divcache = pytest.mark.skipif(
    not (fm.CACHE / "div_SPY.parquet").exists(),
    reason="div_* 캐시가 없다 (원천 전용 트리)")


@needs_divcache
def test_price_row_yf_passes_discrete_divlog(monkeypatch, stub_engine, row):
    """yf 분기는 이산배당 divlog 를 넘기고 qcont 는 넘기지 않는다.

     기본값이 lseg(연속 q)로 바뀌었으므로 이 분기는 명시해서 부른다."""
    monkeypatch.setattr(BMV, "DIV", "yf")
    BMV.price_row(0, row)
    b_kw = stub_engine[-1]                      # 변형 B 호출
    assert b_kw.get("divlog") is not None and "qcont" not in b_kw


def test_price_row_lseg_passes_continuous_q(monkeypatch, stub_engine, row):
    """LSEG 는 연속 q 를 넘기고 이산배당 이벤트 수는 0 으로 기록된다."""
    monkeypatch.setattr(BMV, "DIV", "lseg")
    monkeypatch.setattr(DV, "qcont_lseg", lambda *a, **k: np.array([0.01, 0.02, 0.03], "float32"))
    rec = BMV.price_row(0, row)
    b_kw = stub_engine[-1]
    assert "divlog" not in b_kw and b_kw.get("qcont") is not None
    assert rec["n_div"] == 0
    assert (rec["q_1"], rec["q_2"], rec["q_3"]) == pytest.approx((0.01, 0.02, 0.03))


def test_price_row_lseg_missing_yield_fails(monkeypatch, row):
    """LSEG 수익률이 없는 기초자산은 'lseg div' 로 탈락한다 (yf 와 다른 라벨)."""
    monkeypatch.setattr(BMV, "DIV", "lseg")
    monkeypatch.setattr(DV, "qcont_lseg", lambda *a, **k: None)
    assert BMV.price_row(0, row)["fail"] == "lseg div"


def test_price_row12_schema_drops_variant_a(row12):
    """1·2스타는 변형 B 만 산출한다 — mc_A·sA_*·r_ns 가 없고 nstar 가 더 있다.

     module/star.py 가 이 차이를 전제로 배열을 만든다."""
    rec = BMV.price_row12(0, row12)
    assert "fail" not in rec, rec.get("fail")
    assert set(rec) == set(["i", "item", "nstar", "mc_B", "n_div", "r_bt"]
                           + SB + RHO + QQ + BU)
    assert rec["nstar"] in (1, 2)


def test_price_row12_duplicate_slot_corr_is_exactly_one(row12):
    """중복 슬롯 쌍의 상관은 정확히 1.0 (corr_bd 가 0.999… 로 자르는 것을 되돌린다)."""
    ts = [row12["udl1"], row12["udl2"], row12["udl3"]]
    rec = BMV.price_row12(0, row12)
    for lab, a, b in [("rho_12", 0, 1), ("rho_13", 0, 2), ("rho_23", 1, 2)]:
        if ts[a] == ts[b]:
            assert rec[lab] == 1.0, (lab, rec[lab])
