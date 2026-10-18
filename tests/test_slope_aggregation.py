# -*- coding: utf-8 -*-
"""기울기도 **단일 모형 평균**이어야 한다 — 수준 지표만 고치고 여기를 빼먹었다.

 `generate/build_nb_viz.py` 의 수준 지표는 시드별 지표의 평균으로 고쳤는데(앙상블 아님),
 `generate/eval_all_stars.py` 의 기울기는 여전히 시드 **예측을 평균**한 뒤 기울기를 낸다.
 그러면 `allstar_level.csv` 는 단일 모형, `allstar_slope.csv` 는 5-모형 앙상블이 되어
 README 가 둘을 나란히 쓰면서 같은 대상인 양 말하게 된다. 같은 결함 유형이 한 파일 옆에
 남아 있던 것이다.

 그리고 헤드라인 모형(`ball_dm`)이 전수 횡단면 기울기 도구에 **빠져 있었다** —
 그래서 README 가 "기울기는 전수 횡단면으로" 라는 기록을 두고도 섭동 격자 수치를 써야 했다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm      # noqa: E402

HEADLINE = "ball_dm"


def test_cross_section_covers_the_headline_model():
    """전수 횡단면이 기준 모형을 다뤄야 README 가 그걸 인용할 수 있다."""
    import generate.eval_cross_section as ECS
    prefixes = {m[1] for m in ECS.M}
    assert HEADLINE in prefixes, "전수 횡단면에 기준 모형이 없다: {}".format(sorted(prefixes))


def test_slope_averages_per_seed_not_predictions():
    """시드 예측을 먼저 평균하면 앙상블 기울기가 되어 수준 지표와 다른 것을 잰다."""
    import inspect
    import generate.eval_all_stars as EAS
    src = inspect.getsource(EAS)
    assert "np.mean([don(pre, s, k)" not in src, "기울기가 앙상블 예측으로 계산된다"
    assert "SLOPE_AGG" in src, "집계 방식이 명시돼 있지 않다"
    assert EAS.SLOPE_AGG == "per-seed"
