# -*- coding: utf-8 -*-
"""원천 전용 재현의 판정 — 지수전용 부분집합 대조.

 **왜 부분집합인가.** 가격 계열이 `px_*` 캐시(배당 역조정)에서 `Close_Adj`(액면분할만)로
 바뀌었다. 개별주식은 두 계열이 실제로 다르므로(삼성전자 2015-01-02: 20,419 vs 26,600)
 라벨이 달라지는 것이 **정상**이다. 반면 지수는 두 계열이 거의 같다(^KS200 수익률 상관 0.99994).
 그래서 지수만으로 구성된 상품에서는 라벨이 재현돼야 하고, 그 집합이 판정 대상이다.

 **가중치는 비트 단위로 대조하지 않는다.** 시드가 0~4 로 같아도 라벨이 바뀌었으므로 텐서는
 반드시 다르다. 파일 개수·이름 규약의 일치와 성능 재현으로 본다.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                              # noqa: E402
from tools.verify_raw_run import (TIGHT, IDX, BASE, EXP_MAD, GATE,   # noqa: E402
                                  compare, verdict)

needs_baseline = pytest.mark.skipif(
    not (BASE / "cache" / "mcvar" / "mc_variants_lseg.parquet").exists(),
    reason="구본 스냅샷이 없다 (Task 8 승인 후 삭제됨)")
needs_models = pytest.mark.skipif(
    not any((fm.RESULT / "models").glob("*.pt")),
    reason="신본 가중치가 아직 없다 (노트북 2 미실행)")


def test_tight_is_a_subset_of_idx():
    assert TIGHT < IDX
    assert IDX - TIGHT == {"^HSI", "^STOXX50E"}


def test_subsets_are_disjoint_and_sized():
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet")
    m = d.udl1.isin(IDX) & d.udl2.isin(IDX) & d.udl3.isin(IDX)
    assert 50_000 < int(m.sum()) < 58_790          # 지수전용 56,162건 전후


@needs_baseline
def test_compare_reports_three_subsets():
    T = compare()
    assert list(T.subset) == ["TIGHT", "IDX", "OTHER"]
    assert T.n.sum() == len(pd.read_parquet(fm.DATA / "els3_dataset.parquet"))


@needs_baseline
def test_compare_only_uses_items_present_in_both():
    """신본 전용 123건을 섞으면 상관이 의미를 잃는다 — 공통 항목만 쓰고 개수를 밝힌다."""
    T = compare()
    assert (T.n_cmp <= T.n).all()
    assert T.n_cmp.sum() < T.n.sum()
    assert T.n_only_new.sum() > 0                  # 새로 살아난 상품을 따로 보고한다


@needs_baseline
def test_compare_columns_are_pinned():
    T = compare()
    assert list(T.columns) == ["subset", "n", "n_cmp", "n_only_new", "mc_corr", "mc_mad",
                               "mc_mad_exp", "mc_bias", "px_n", "px_mad", "px_bias",
                               "r2_old", "r2_new", "mape_old", "mape_new"]


@needs_baseline
def test_label_diff_cannot_test_reproduction():
    """구본과 신본은 공통난수가 아니다 — mc_seed 가 행 번호라 123건 유입으로 98% 가 바뀌었다.

     그래서 mc_mad 는 '재현 실패'가 아니라 독립 MC 잡음 두 개의 합이고, 기대값이 EXP_MAD 다.
     판정은 공통난수로 잡음이 상쇄된 px_mad 로 해야 한다."""
    o = pd.read_parquet(BASE / "datasets" / "els3_dataset.parquet",
                        columns=["item", "mc_seed"]).set_index("item")
    n = pd.read_parquet(fm.DATA / "els3_dataset.parquet",
                        columns=["item", "mc_seed"]).set_index("item")
    J = o.join(n, how="inner", lsuffix="_o", rsuffix="_n")
    assert (J.mc_seed_o == J.mc_seed_n).mean() < 0.05          # 실측 1.8%
    assert 6.0 < EXP_MAD < 8.5                                  # 40k + 100k 잡음 합 7.18원


@needs_baseline
def test_px_effect_is_noise_cancelled():
    """공통난수 측정에서 TIGHT 지수의 가격계열 효과는 사실상 0 이어야 한다."""
    T = compare().set_index("subset")
    assert T.at["TIGHT", "px_mad"] < 1.0
    assert T.at["OTHER", "px_mad"] > T.at["IDX", "px_mad"] > T.at["TIGHT", "px_mad"]


@needs_baseline
def test_verdict_judges_on_px_mad_not_label_diff():
    T = compare()
    V = verdict(T)
    assert set(V) == {"TIGHT", "IDX", "OTHER"}
    assert V["TIGHT"][0] is True                                # 0.10원 << 5원
    assert V["OTHER"][0] is None                                # 기준 없음
    assert set(GATE) == {"TIGHT", "IDX"}


@needs_models
def test_weight_inventory_matches_baseline():
    """개수와 이름 규약은 일치해야 한다 (텐서 값은 라벨이 달라 다를 수밖에 없다)."""
    old = {p.name for p in (BASE / "models").glob("*")}
    new = {p.name for p in (fm.RESULT / "models").glob("*")}
    assert new == old
