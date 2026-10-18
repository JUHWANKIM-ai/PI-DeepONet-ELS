# -*- coding: utf-8 -*-
"""보고하는 성능 지표는 **단일 모형 평균**이어야 한다 — 표마다 다른 것을 재면 안 된다.

 두 가지 집계가 섞여 있었다.
   allstar_level  시드별 지표를 낸 뒤 평균          = 단일 모형 평균 성능
   cat1 / cat2    시드 예측을 평균한 뒤 지표        = 5-모형 **앙상블** 성능

 둘 다 타당한 양이지만 다른 것이고, 그림 제목은 양쪽 다 "5 seeds ± 1 SE" 라 구분되지 않았다.
 같은 3스타·같은 n=24,307 에서 R² 가 0.9576(cat1) 과 0.9460(allstar) 으로 갈렸다.
 더 나쁜 것은 cat1 막대가 앙상블인데 오차막대는 단일 시드들의 산포여서, 한 패널 안에서
 서로 다른 대상을 그리고 있었다는 점이다.

 사용자 결정(A): 보고 지표를 전부 단일 모형 평균으로 통일한다. 논문의 다른 표가 모두 그쪽이고,
 앙상블을 섞으면 7종 비교표와 스타별 표가 서로 다른 것을 재게 된다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm      # noqa: E402

TOL = 2e-3          # 폴드 조립 경로가 달라 완전 동일하지는 않다. 집계 방식 차이(0.012)보다 훨씬 작다.

pytestmark = pytest.mark.skipif(
    not (fm.stat("cat1_level").exists() and fm.stat("allstar_level").exists()),
    reason="성능 표가 없다 (노트북 3 미실행)")


def test_star_level_matches_the_seven_model_table():
    """같은 3스타·같은 두 모형을 두 표가 같은 값으로 보고해야 한다."""
    c = pd.read_csv(fm.stat("cat1_level")).set_index(["star", "model"])
    a = pd.read_csv(fm.stat("allstar_level")).set_index(["group", "model"])
    pairs = [("trained 1·2·3-star", "Direct+m (1·2·3)"),
             ("trained 3-star only", "Direct+m (3star only)")]
    for star, group in [("1-star", "1star"), ("2-star", "2star"), ("3-star", "3star")]:
        for cm, am in pairs:
            got, want = c.loc[(star, cm), "R2"], a.loc[(group, am), "R2"]
            assert abs(got - want) < TOL, \
                "{} {}: cat1 {:.4f} vs allstar {:.4f}".format(star, cm, got, want)


def test_generator_averages_metrics_not_predictions():
    """시드 예측 평균(앙상블)으로 되돌아가면 두 표가 다시 갈린다."""
    import inspect
    import tools.build_nb_viz as BNV
    src = inspect.getsource(BNV)
    assert "mets_seed(" in src, "단일 모형 평균 집계를 쓰지 않는다"
    assert "r = mets(idx, p)" not in src, "cat1 이 앙상블 예측으로 지표를 낸다"


def test_error_bars_describe_the_same_object_as_the_bars():
    """막대가 단일 모형 평균이면 오차막대(시드 SE)와 같은 대상을 잰다."""
    c = pd.read_csv(fm.stat("cat1_level"))
    assert {"R2_se", "MAPE_se"} <= set(c.columns)
    assert (c.R2_se > 0).all()


def test_unified_training_buys_coverage_not_three_star_accuracy():
    """README 주장의 근거 — 통합 학습의 이점은 3스타 정확도가 아니라 **커버리지**다.

     **순서를 못박지 않는다.** 3스타에서 어느 쪽이 앞서는지는 라벨 잡음에 따라 뒤집힌다 —
     직전 라벨에서는 3스타 전용이 0.9557 대 0.9460 으로 앞섰고, 유니버스를 59,037건으로
     늘린 뒤에는 0.9540 대 0.9534 로 구별되지 않는다(0.09 결합 SE). 재현 가능한 사실은
     "3스타에서는 둘이 비슷하고, 1·2스타에서는 통합이 압도한다" 이지 특정 순서가 아니다.
     한때 나는 그 순서를 사실로 적었고, 라벨이 바뀌자 거짓이 됐다."""
    c = pd.read_csv(fm.stat("cat1_level")).set_index(["star", "model"])
    u3, o3 = c.loc[("3-star", "trained 1·2·3-star")], c.loc[("3-star", "trained 3-star only")]
    gap = abs(u3.R2 - o3.R2) / np.hypot(u3.R2_se, o3.R2_se)
    assert gap < 3, "3스타에서 한쪽이 유의하게 앞선다 ({:.1f} SE) — 주장을 다시 써야 한다".format(gap)
    for star in ("1-star", "2-star"):
        u, o = c.loc[(star, "trained 1·2·3-star")], c.loc[(star, "trained 3-star only")]
        assert u.R2 - o.R2 > 5 * np.hypot(u.R2_se, o.R2_se), \
            "{}: 통합의 우위가 사라졌다".format(star)
