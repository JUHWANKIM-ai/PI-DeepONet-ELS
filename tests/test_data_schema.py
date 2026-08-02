import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest

from util import file_manager as fm
from module import data as DA
from module.schedule import NSTRK, LZ_BARR_NEUTRAL, LZ_PMT_NEUTRAL


def test_contract_length_and_order():
    assert len(DA.CONTRACT) == 4 * NSTRK + 3
    assert DA.CONTRACT[0] == "strk_0"
    assert DA.CONTRACT[NSTRK] == "pmt_0"
    assert DA.CONTRACT[2 * NSTRK] == "lz_barr_0"
    assert DA.CONTRACT[3 * NSTRK] == "lz_pmt_0"
    assert DA.CONTRACT[-3:] == ["BARR_1/100", "ANL_RTRN/100", "TENOR"]


def test_contract_contains_physics_columns():
    """load() 가 D.iBARR/iCOUPON/iTEN/iKlast 를 CONTRACT.index() 로 잡으므로 이름이 있어야 한다."""
    for c in ("BARR_1/100", "ANL_RTRN/100", "TENOR", f"strk_{NSTRK - 1}"):
        assert c in DA.CONTRACT, f"CONTRACT 에 {c} 없음"


def test_structure_columns_are_categorical_in_ml():
    assert "OPT_TYPE" in DA.CAT
    assert "KNCK_IN_YN" in DA.CAT


def test_neutral_fill_replaces_nan_lizard():
    df = pd.DataFrame({"lz_barr_0": [np.nan, 0.6], "lz_pmt_0": [np.nan, 0.024]})
    out = DA.fill_lizard_neutral(df.copy())
    assert out.loc[0, "lz_barr_0"] == LZ_BARR_NEUTRAL
    assert out.loc[0, "lz_pmt_0"] == LZ_PMT_NEUTRAL
    assert out.loc[1, "lz_barr_0"] == 0.6          # 실제값은 건드리지 않는다
    assert out.loc[1, "lz_pmt_0"] == 0.024


def test_deeponet_csv_has_no_nan_in_contract():
    p = fm.dataset("deeponet")
    if not p.exists():
        pytest.skip("deeponet.csv 없음")
    don = pd.read_csv(p, encoding=DA.CSV_ENC)
    if "pmt_0" not in don.columns:
        pytest.skip("아직 새 스키마로 재생성되지 않음")
    assert don[DA.CONTRACT].isna().sum().sum() == 0


def test_loaded_tensors_have_expected_widths():
    p = fm.dataset("deeponet")
    if not p.exists():
        pytest.skip("deeponet.csv 없음")
    don = pd.read_csv(p, encoding=DA.CSV_ENC, nrows=2)
    if "pmt_0" not in don.columns:
        pytest.skip("아직 새 스키마로 재생성되지 않음")
    from util import utils
    D = DA.load(utils.load_config())
    assert D.CON.shape[1] == len(DA.CONTRACT)
    assert D.DON.shape[1] == len(DA.UC) + len(DA.VOLCORR) + len(DA.CONTRACT)
    assert D.VC.shape[1] == 7 and D.CURVE.shape[1] == 10
    assert DA.CONTRACT[D.iBARR] == "BARR_1/100"
    assert DA.CONTRACT[D.iCOUPON] == "ANL_RTRN/100"
    assert DA.CONTRACT[D.iTEN] == "TENOR"
    assert DA.CONTRACT[D.iKlast] == f"strk_{NSTRK - 1}"
