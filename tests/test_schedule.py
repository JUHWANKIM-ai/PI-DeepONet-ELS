import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from module.schedule import build_schedules, NSTRK, SCHED_COLS


COLS = ["ITEM_CD", "SCHD_TYPE", "SEQ", "STRK_1", "PMT_1", "BARR_1",
        "LZRD_BARR", "LZRD_PMT", "LZRD_TERM"]


def _sc(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_pads_to_nstrk_with_forward_fill():
    out = build_schedules(_sc([
        ("A", 1, 1, 90.0, 0.03, 50.0, np.nan, np.nan, np.nan),
        ("A", 1, 2, 85.0, 0.06, 50.0, np.nan, np.nan, np.nan),
    ]))
    assert out.loc["A", "nobs"] == 2
    assert out.loc["A", "strk_0"] == 0.90          # 퍼센트 -> 소수
    assert out.loc["A", "strk_1"] == 0.85
    assert out.loc["A", "strk_11"] == 0.85         # 마지막값 forward-fill
    assert out.loc["A", "pmt_0"] == 0.03           # PMT_1 은 이미 소수 (나누지 않음)
    assert out.loc["A", "pmt_11"] == 0.06
    assert bool(out.loc["A", "sched_ok"]) is True


def test_lizard_lands_on_its_own_seq_only():
    out = build_schedules(_sc([
        ("B", 1, 1, 90.0, 0.03, 50.0, np.nan, np.nan, np.nan),
        ("B", 1, 2, 85.0, 0.06, 50.0, 60.0, 0.024, "FROM_ISU"),
        ("B", 1, 3, 80.0, 0.09, 50.0, np.nan, np.nan, np.nan),
    ]))
    assert np.isnan(out.loc["B", "lz_barr_0"])
    assert out.loc["B", "lz_barr_1"] == 0.60       # 퍼센트 -> 소수
    assert out.loc["B", "lz_pmt_1"] == 0.024       # 이미 소수
    assert np.isnan(out.loc["B", "lz_barr_2"])
    assert np.isnan(out.loc["B", "lz_barr_11"])    # 리자드는 forward-fill 하지 않는다
    assert bool(out.loc["B", "lz_from_prev"]) is False


def test_from_prev_is_flagged():
    out = build_schedules(_sc([
        ("C", 1, 1, 90.0, 0.03, 50.0, 60.0, 0.02, "FROM_PREV"),
        ("C", 1, 2, 85.0, 0.06, 50.0, np.nan, np.nan, np.nan),
    ]))
    assert bool(out.loc["C", "lz_from_prev"]) is True


def test_more_than_nstrk_observations_is_not_ok():
    rows = [("D", 1, j + 1, 90.0 - j, 0.01 * (j + 1), 50.0, np.nan, np.nan, np.nan)
            for j in range(NSTRK + 1)]
    out = build_schedules(_sc(rows))
    assert bool(out.loc["D", "sched_ok"]) is False
    assert out.loc["D", "sched_drop"] == "nobs>12"


def test_missing_strike_or_pmt_is_not_ok():
    out = build_schedules(_sc([
        ("E", 1, 1, np.nan, 0.03, 50.0, np.nan, np.nan, np.nan),
        ("E", 1, 2, 85.0, 0.06, 50.0, np.nan, np.nan, np.nan),
        ("F", 1, 1, 90.0, np.nan, 50.0, np.nan, np.nan, np.nan),
        ("F", 1, 2, 85.0, 0.06, 50.0, np.nan, np.nan, np.nan),
    ]))
    assert out.loc["E", "sched_drop"] == "strk_na"
    assert out.loc["F", "sched_drop"] == "pmt_na"


def test_too_few_observations_is_not_ok():
    out = build_schedules(_sc([("G", 1, 1, 90.0, 0.03, 50.0, np.nan, np.nan, np.nan)]))
    assert bool(out.loc["G", "sched_ok"]) is False
    assert out.loc["G", "sched_drop"] == "nobs<2"


def test_schd_type_2_rows_are_ignored():
    out = build_schedules(_sc([
        ("H", 1, 1, 90.0, 0.03, 50.0, np.nan, np.nan, np.nan),
        ("H", 1, 2, 85.0, 0.06, 50.0, np.nan, np.nan, np.nan),
        ("H", 2, 1, 10.0, 0.99, 10.0, np.nan, np.nan, np.nan),
    ]))
    assert out.loc["H", "nobs"] == 2
    assert out.loc["H", "strk_0"] == 0.90


def test_non_monotone_cumulative_payment_is_flagged():
    """누적 지급률은 회차가 갈수록 줄 수 없다. raw 에 1회차가 만기값인 오류가 실제로 있다."""
    out = build_schedules(_sc([
        # 정상: 0.03 -> 0.06 -> 0.09
        ("OK", 1, 1, 90.0, 0.03, 50.0, np.nan, np.nan, np.nan),
        ("OK", 1, 2, 85.0, 0.06, 50.0, np.nan, np.nan, np.nan),
        ("OK", 1, 3, 80.0, 0.09, 50.0, np.nan, np.nan, np.nan),
        # 오류: 1회차가 만기 지급률(0.201)
        ("BAD", 1, 1, 80.0, 0.2010, np.nan, np.nan, np.nan, np.nan),
        ("BAD", 1, 2, 80.0, 0.0670, np.nan, np.nan, np.nan, np.nan),
        ("BAD", 1, 3, 85.0, 0.2010, 50.0, np.nan, np.nan, np.nan),
    ]))
    assert bool(out.loc["OK", "pmt_nonmono"]) is False
    assert bool(out.loc["BAD", "pmt_nonmono"]) is True
    assert bool(out.loc["BAD", "sched_ok"]) is True          # 제외가 아니라 플래그만 (선형 대체는 build_source)


def test_sched_cols_shape():
    assert len(SCHED_COLS) == 4 * NSTRK
    out = build_schedules(_sc([
        ("A", 1, 1, 90.0, 0.03, 50.0, np.nan, np.nan, np.nan),
        ("A", 1, 2, 85.0, 0.06, 50.0, np.nan, np.nan, np.nan),
    ]))
    for c in SCHED_COLS:
        assert c in out.columns
