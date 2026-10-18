# -*- coding: utf-8 -*-
"""헤드라인 그림 — 3스타에서 기준 모형(Direct+f+m)과 표형 벤치마크(XGBoost) 비교.

 README 가 내세우는 그림 한 장이다. 규약 세 가지를 지켜야 한다.
   1. 패널은 **R² 와 MAPE 두 개**만 (MAE·RMSE 등은 표로만 보고한다)
   2. 그림 안 글자는 **전부 영어**
   3. 막대는 **단일 모형 5시드 평균**, 오차막대는 같은 시드들의 SE
      (앙상블 예측으로 지표를 내면 막대와 오차막대가 서로 다른 대상이 된다)

 수치는 result/statistics/allstar_level.csv 의 group=3star 행에서 그대로 온다 —
 그림이 표와 다른 값을 말하면 안 된다.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm      # noqa: E402

NAME = "headline_3star"
MODELS = ("Direct+f+m (1·2·3)", "XGBoost (1·2·3)")


def test_generator_declares_the_two_models_and_two_panels():
    import inspect
    import tools.eval_all_stars as EAS
    src = inspect.getsource(EAS)
    assert "headline_3star" in src, "헤드라인 그림 생성기가 없다"
    assert "HEADLINE" in src
    assert EAS.HEADLINE == MODELS
    assert EAS.HEADLINE_PANELS == ("R2", "MAPE"), "패널은 R²·MAPE 두 개로 고정한다"


@pytest.mark.skipif(not fm.stat("allstar_level").exists(), reason="성능 표 없음 (노트북 3 미실행)")
def test_source_rows_exist_for_the_figure():
    L = pd.read_csv(fm.stat("allstar_level"))
    g = L[(L.group == "3star") & (L.model.isin(MODELS))]
    assert len(g) == 2, "3스타 두 모형 행이 모두 있어야 한다"
    assert {"R2_se", "MAPE_se"} <= set(g.columns)
    d = g.set_index("model")
    assert d.at[MODELS[0], "R2"] > d.at[MODELS[1], "R2"], "기준 모형이 벤치마크보다 낮다"


@pytest.mark.skipif(not fm.image(NAME).exists(), reason="그림 없음 (노트북 3 미실행)")
def test_figure_is_produced():
    assert fm.image(NAME).stat().st_size > 20_000


def test_allstar_panels_use_r2_and_mape_not_mae():
    """7종 통합 그림의 성능 패널도 R²·MAPE 두 지표로 고정한다.

     종전에는 1·2·3스타 패널이 **MAE 막대**였다. 스타별로 가격대가 달라 원 단위 오차가
     직관적이긴 하지만, 성능 히스토그램은 R²·MAPE 로 통일하고 MAE·RMSE 등은 표로만 낸다
     (`allstar_level.csv` 에 여섯 지표가 모두 있다). 기울기 패널은 비율이라 그대로 둔다."""
    import inspect
    import tools.eval_all_stars as EAS
    src = inspect.getsource(EAS)
    assert EAS.STAR_PANELS == ("R2", "MAPE")
    assert "OOS MAE [KRW]" not in src, "스타별 패널에 MAE 막대가 남아 있다"
    assert "mean |1 - slope / MC slope|" in src, "기울기 패널은 유지한다"
