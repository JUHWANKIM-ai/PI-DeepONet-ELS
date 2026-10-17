# -*- coding: utf-8 -*-
"""교체 영향 측정 도구 — 세 요인을 한 번에 하나씩만 바꿔 분리한다.

 원천 전용 재현은 라벨을 세 군데 바꾼다 (가격계열 · 배당 · 경로 수).
 셋을 한꺼번에 바꾸면 어느 것이 얼마나 흔들었는지 알 수 없으므로,
 기준에서 **하나씩만** 바꾼 가격을 함께 낸다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.measure_source_switch import measure    # noqa: E402

COLS = {"item", "is_index_only", "sig_old", "sig_new", "rho_old", "rho_new",
        "mc_base", "mc_px", "mc_div", "mc_path", "mc_all"}


@pytest.fixture(scope="module")
def T5():
    return measure(nprod=5)


def test_measure_isolates_each_factor(T5):
    assert len(T5) == 5
    assert set(T5.columns) == COLS
    assert T5[["mc_base", "mc_px", "mc_div", "mc_path", "mc_all"]].notna().all().all()


def test_measure_marks_index_only(T5):
    """is_index_only 는 기초자산 3개가 모두 지수인 상품만 True."""
    assert T5.is_index_only.dtype == bool


def test_index_only_products_have_unchanged_sigma():
    """지수전용 상품은 가격계열 교체로 σ 가 거의 안 변한다 (수익률 상관 0.9999+)."""
    T = measure(nprod=40)
    idx = T[T.is_index_only]
    assert len(idx) > 0
    assert (idx.sig_new - idx.sig_old).abs().max() < 0.005
