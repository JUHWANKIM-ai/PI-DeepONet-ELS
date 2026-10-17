# -*- coding: utf-8 -*-
"""노트북 2(생산)와 노트북 3(소비)의 가중치 계약.

 노트북 3 은 `eval_all_stars` · `eval_cross_section` 을 부르고, 그 둘은 가중치를 **불러올 뿐
 학습하지 않는다**. 그래서 노트북 2 가 7종을 전부 학습하지 않으면 노트북 3 이 깨진다.

 원천 전용 실행에서 실제로 그랬다: 노트북 2 가 `ball_d`·`ball3_d` 두 종(40개)만 만들어
 구본의 140개와 어긋났고, 빠진 다섯 종에 **헤드라인 비교 그림의 막대 5개**가 들어 있었다.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                  # noqa: E402

SEEDS, FOLDS = 5, 4


def consumed():
    """두 평가 도구가 불러오는 접두사 집합."""
    import tools.eval_all_stars as EAS
    import tools.eval_cross_section as ECS
    return {m[1] for m in EAS.M} | {m[1] for m in ECS.M}


def test_notebook2_trains_every_variant_the_evals_consume():
    import inspect
    import tools.build_nb_training as BNT
    src = inspect.getsource(BNT)
    missing = [t for t in sorted(consumed()) if '"{}"'.format(t) not in src]
    assert missing == [], "노트북 2 가 학습하지 않는데 노트북 3 이 읽는다: {}".format(missing)


def test_seven_variants_are_expected():
    """개수가 줄면 비교 그림의 막대가 조용히 사라진다."""
    assert len(consumed()) == 7


def test_xgb_variants_are_pickles_and_others_are_torch():
    """eval_all_stars 는 접두사가 _xgb 로 끝나면 .pkl, 아니면 .pt 를 읽는다."""
    xgb = {t for t in consumed() if t.endswith("_xgb")}
    assert xgb == {"ball_xgb", "ball3_xgb"}


@pytest.mark.skipif(not any((fm.RESULT / "models").glob("*")),
                    reason="가중치가 없다 (노트북 2 미실행)")
def test_weight_count_matches_the_contract():
    got = {p.name for p in (fm.RESULT / "models").glob("*")}
    want = set()
    for t in consumed():
        ext = ".pkl" if t.endswith("_xgb") else ".pt"
        tag = t if t.endswith("_xgb") else t + "_deeponet"
        for s in range(SEEDS):
            for k in range(FOLDS):
                want.add("{}_anchor_seed{}_fold{}{}".format(tag, s, k, ext))
    assert len(want) == SEEDS * FOLDS * 7 == 140
    assert got == want
