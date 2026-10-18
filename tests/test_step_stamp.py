# -*- coding: utf-8 -*-
"""노트북 1 의 각 단계는 **어떤 라벨로 만들어졌는지**로 완료를 판단해야 한다.

 같은 결함을 이 세션에서 네 번 고쳤다 — 데이터셋(`MC` 전부 NaN), 가중치(파일 존재),
 3스타 라벨(반쪽 파일), 그리고 섭동 격자·요인분해. 매번 다른 임시방편을 붙였다.
 산출물마다 다른 표식을 쓰면 다음 산출물에서 또 생긴다.

 공통 메커니즘: 각 단계가 끝나면 그 시점의 **라벨 출처 스탬프**를 `<path>.stamp.json` 에 남기고,
 다음 실행은 현재 스탬프와 비교한다. "파일이 있는가" 가 아니라 "무엇으로 만든 파일인가" 다.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                  # noqa: E402
import tools.build_mc_variants as BMV                # noqa: E402


def test_step_stale_when_no_mark(tmp_path):
    f = tmp_path / "out.csv"
    f.write_text("x", encoding="utf-8")
    assert BMV.step_stale(f) is True                  # 표식이 없으면 낡은 것으로 본다


def test_step_stale_is_false_right_after_marking(tmp_path, monkeypatch):
    lab = tmp_path / "lab.parquet"
    BMV.write_labeled(pd.DataFrame({"item": ["A"], "mc_B": [0.9]}), lab)
    monkeypatch.setattr(BMV, "OUT", tmp_path)
    monkeypatch.setattr(BMV, "OUT_NAME", "lab.parquet")
    f = tmp_path / "out.csv"
    f.write_text("x", encoding="utf-8")
    BMV.mark_step(f)
    assert BMV.step_stale(f) is False


def test_step_stale_when_the_label_changed(tmp_path, monkeypatch):
    lab = tmp_path / "lab.parquet"
    BMV.write_labeled(pd.DataFrame({"item": ["A"], "mc_B": [0.9]}), lab)
    monkeypatch.setattr(BMV, "OUT", tmp_path)
    monkeypatch.setattr(BMV, "OUT_NAME", "lab.parquet")
    f = tmp_path / "out.csv"
    f.write_text("x", encoding="utf-8")
    BMV.mark_step(f)
    BMV.write_labeled(pd.DataFrame({"item": ["A", "B"], "mc_B": [0.9, 1.0]}), lab)   # 유니버스 변경
    assert BMV.step_stale(f) is True


def test_missing_file_is_stale(tmp_path):
    assert BMV.step_stale(tmp_path / "nope.csv") is True


def test_notebook1_uses_the_shared_step_check():
    """단계마다 다른 임시방편을 붙이면 다음 산출물에서 또 생긴다."""
    import inspect
    import tools.build_nb_pricer as BNP
    src = inspect.getsource(BNP)
    assert "step_stale" in src and "mark_step" in src
    assert 'q.name == "mc_variants.parquet"' not in src, "산출물별 임시방편이 남아 있다"


@pytest.mark.skipif(not fm.stat("mc_check_allstar").exists(), reason="격자 산출물 없음")
def test_shipped_grid_and_decomp_match_the_current_label():
    """격자·요인분해도 mc_seed 를 쓰므로 라벨이 바뀌면 낡는다."""
    for name in ("mc_check_allstar", "mc_boot_curve_200"):
        assert not BMV.step_stale(fm.stat(name)), "{} 가 낡았다".format(name)
