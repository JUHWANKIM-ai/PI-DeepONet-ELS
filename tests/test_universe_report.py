import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import pytest

from util import file_manager as fm
from module import data as DA


@pytest.fixture(scope="module")
def ml():
    p = fm.dataset("ml")
    if not p.exists():
        pytest.skip("ml.csv 없음")
    d = pd.read_csv(p, encoding=DA.CSV_ENC)
    if "OPT_TYPE" not in d.columns:
        pytest.skip("아직 새 스키마로 재생성되지 않음")
    return d


def test_all_four_structures_present(ml):
    got = set(zip(ml["OPT_TYPE"].astype(str), ml["KNCK_IN_YN"].astype(int)))
    assert got == {("STEP", 1), ("STEP", 0), ("LIZARD", 1), ("LIZARD", 0)}


def test_no_ki_products_carry_barrier_one(ml):
    noki = ml[ml["KNCK_IN_YN"].astype(int) == 0]
    assert len(noki) > 0
    assert (noki["BARR_1/100"] == 1.0).all()


def test_step_products_have_neutral_lizard_columns(ml):
    don = pd.read_csv(fm.dataset("deeponet"), encoding=DA.CSV_ENC)
    step = don.loc[ml["OPT_TYPE"].astype(str) == "STEP"]
    assert (step[[f"lz_barr_{j}" for j in range(12)]] == 1.2).all().all()
    assert (step[[f"lz_pmt_{j}" for j in range(12)]] == 0.0).all().all()


def test_lizard_products_have_a_real_barrier_somewhere(ml):
    don = pd.read_csv(fm.dataset("deeponet"), encoding=DA.CSV_ENC)
    lz = don.loc[ml["OPT_TYPE"].astype(str) == "LIZARD"]
    cols = [f"lz_barr_{j}" for j in range(12)]
    assert (lz[cols] < 1.2).any(axis=1).all(), "리자드 상품인데 실제 배리어가 하나도 없다"


def test_mc_is_finite_and_in_plausible_range(ml):
    assert ml["MC"].notna().all()
    assert ml["MC"].between(0.3, 1.15).all(), (ml["MC"].min(), ml["MC"].max())
