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

from util import file_manager as fm                 # noqa: E402
from tools.measure_source_switch import measure    # noqa: E402

# 이 도구는 **두 가격계열을 한 번에 둘 다** 써야 요인을 분리할 수 있다 — 구 계열인 px_* 캐시가
# 반드시 필요하다. 원천 전용 트리에서는 그 캐시가 격리돼 있으므로 모듈 전체를 건너뛴다.
# (측정 자체는 Task 1 에서 캐시가 있을 때 돌렸고, 결과는 source_switch_impact.csv 에 남아 있다.)
pytestmark = pytest.mark.skipif(
    not (fm.CACHE / "px_005930_KS.parquet").exists(),
    reason="px_* 캐시가 없다 (원천 전용 트리)")

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


def test_measurement_is_not_vacuous(T5):
    """구 계열과 신 계열이 실제로 달라야 요인 분리가 의미를 갖는다.

     `PX_SOURCE` 기본값이 cache -> underlying 으로 바뀐 뒤, 이 도구의 '구 계열'(MK.returns())이
     신 계열과 같아져 측정이 통째로 공허해졌다. 그러면 px_mad 가 0.00 이 되고
     tools/verify_raw_run.verdict() 가 **공허한 측정 위에서 '통과'** 를 찍는다 —
     실패 방향이 '통과' 라서 눈에 띄지 않는다."""
    assert (T5.sig_old != T5.sig_new).any(), "구/신 변동성이 전부 같다 — 구 계열을 못 읽고 있다"
    assert (T5.mc_base != T5.mc_px).any(), "가격계열 교체 효과가 전부 0 이다 — 측정이 공허하다"
