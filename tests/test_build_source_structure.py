import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest

from module.build_source import STRUCTS, encode_barrier, structure_mask


def test_structs_covers_two_by_two():
    assert set(STRUCTS) == {("STEP", 1), ("STEP", 0), ("LIZARD", 1), ("LIZARD", 0)}


def test_encode_barrier_ki_divides_by_100():
    assert encode_barrier(1, 50.0) == 0.5
    assert encode_barrier(1, 45.0) == 0.45


def test_encode_barrier_no_ki_is_one_even_if_barrier_present():
    # 노낙인인데 BARR_1 이 붙은 상품이 1건 존재 -> KNCK_IN_YN 을 신뢰한다
    assert encode_barrier(0, 60.0) == 1.0
    assert encode_barrier(0, np.nan) == 1.0


def test_encode_barrier_ki_without_barrier_is_nan():
    assert np.isnan(encode_barrier(1, np.nan))


def test_structure_mask_keeps_four_structures_only():
    ac = pd.DataFrame({
        "ITEM_CD": ["a", "b", "c", "d", "e", "f"],
        "OPT_TYPE": ["STEP", "LIZARD", "STEP", "LIZARD", "STEP", "STEP"],
        "KNCK_IN_YN": [1, 0, 0, 1, 1, 1],
        "CUR_CD": ["KRW", "KRW", "KRW", "KRW", "USD", "KRW"],
        "fair": [0.9, 0.9, 0.9, 0.9, 0.9, 0.5],       # f 는 공정가 범위 밖
        "tenor": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    })
    m = structure_mask(ac, pd.Index(["a", "b", "c", "d", "e", "f"]))
    assert list(ac.loc[m, "ITEM_CD"]) == ["a", "b", "c", "d"]


def test_structure_mask_requires_three_underlyings():
    ac = pd.DataFrame({
        "ITEM_CD": ["a", "b"],
        "OPT_TYPE": ["STEP", "STEP"],
        "KNCK_IN_YN": [1, 1],
        "CUR_CD": ["KRW", "KRW"],
        "fair": [0.9, 0.9],
        "tenor": [3.0, 3.0],
    })
    m = structure_mask(ac, pd.Index(["a"]))       # b 는 3-star 아님
    assert list(ac.loc[m, "ITEM_CD"]) == ["a"]


def test_built_source_has_structure_and_underlying_columns():
    """빌드 산출물 스키마 — Task 2 Step 6 실행 후에만 검사(느린 빌드는 여기서 하지 않는다)."""
    from util import file_manager as fm
    if not fm.source().exists():
        pytest.skip("els3_dataset.parquet 없음")
    d = pd.read_parquet(fm.source())
    if "opt_type" not in d.columns:
        pytest.skip("아직 4구조로 재빌드되지 않음")
    for c in ("opt_type", "ki_yn", "udl1", "udl2", "udl3", "udl_key", "nobs",
              "strk_0", "pmt_0", "lz_barr_0", "lz_pmt_0"):
        assert c in d.columns, f"missing {c}"
    assert (d["udl1"] <= d["udl2"]).all() and (d["udl2"] <= d["udl3"]).all()   # 정렬 규약
    assert d["udl_key"].str.count(r"\|").eq(2).all()
