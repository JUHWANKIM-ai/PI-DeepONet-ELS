import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest

from util import file_manager as fm
from module import mc as MC


@pytest.fixture(scope="module")
def src():
    # Task 2 Step 5 에서 뜬 '재빌드 이전' 스냅샷 — 현재 parquet 은 mc 가 아직 없다(Task 5 에서 채움)
    p = fm.CACHE / "_prev_els3.parquet"
    assert p.exists(), "Task 2 Step 5 의 스냅샷(_prev_els3.parquet)이 없다"
    return pd.read_parquet(p).sort_values("isu_ord").reset_index(drop=True)


@pytest.fixture(scope="module")
def mk():
    return MC.load_market()


def test_prepare_one_applies_calibration_multiplier(mk, src):
    r = src.loc[0]
    p = MC.prepare_one(mk, None, r["item"], int(r["isu_ord"]), float(r["tenor"]),
                       float(r["sig_eff"]))
    assert p is not None
    assert p["k"] == 1.0                                   # kmap=None -> 미보정
    np.testing.assert_allclose(p["sigs"], p["audit"][:3], rtol=1e-12)
    assert len(p["audit"]) == 6


def test_prepare_one_honours_explicit_strikes(mk, src):
    r = src.loc[0]
    p = MC.prepare_one(mk, None, r["item"], int(r["isu_ord"]), float(r["tenor"]),
                       float(r["sig_eff"]), strikes=[0.9, 0.8, 0.7])
    assert p["strikes"] == [0.9, 0.8, 0.7]


def test_price_one_returns_nine_fields(mk, src):
    r = src.loc[0]
    v = MC.price_one(mk, None, r["item"], int(r["isu_ord"]), float(r["B"]), float(r["coupon"]),
                     float(r["tenor"]), float(r["sig_eff"]), n=2000)
    assert len(v) == len(MC.MC_COLS) == 9


def test_price_one_reproduces_dataset_mc_bitwise(mk, src):
    for i in range(5):
        r = src.loc[i]
        v = MC.price_one(mk, None, r["item"], int(r["isu_ord"]), float(r["B"]),
                         float(r["coupon"]), float(r["tenor"]), float(r["sig_eff"]))
        assert abs(v[0] - float(r["mc"])) < 1e-6, f"row {i}: {v[0]} vs {r['mc']}"
