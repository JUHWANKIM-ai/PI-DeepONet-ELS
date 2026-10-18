# -*- coding: utf-8 -*-
"""라벨 파일의 스키마는 **실패 유무에 따라 달라지면 안 된다.**

 `price_row` 는 실패했을 때만 `rec["fail"]` 을 넣는다. 그래서 전건 성공하면 combine() 이
 `fail` 열 **없는** 프레임을 쓰고, 그 뒤 `fix()` 의 `R["fail"].isna()` 가 KeyError 로 죽는다.

 실제로 그랬다: CSI 300·EURO STOXX Banks 배당 프록시를 고쳐 59,037건이 전건 성공하자
 (`combined 59,037 | fail 0`) 137분짜리 라벨 산출 직후에 노트북 1 이 터졌다.
 **성공할수록 깨지는** 코드였다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                  # noqa: E402
import tools.build_mc_variants as BMV                # noqa: E402

DONE_COLS = {"fail", "px_lag_days", "stale_px"}


def test_ensure_fail_adds_the_column_when_nothing_failed():
    R = pd.DataFrame({"i": [0, 1], "item": ["A", "B"], "mc_B": [0.9, 1.0]})
    out = BMV.ensure_fail(R)
    assert "fail" in out.columns
    assert out["fail"].isna().all()


def test_ensure_fail_keeps_existing_values():
    R = pd.DataFrame({"i": [0, 1], "item": ["A", "B"], "fail": [None, "lseg div"]})
    out = BMV.ensure_fail(R)
    assert out["fail"].isna().tolist() == [True, False]     # pandas 가 None 을 NaN 으로 바꾼다
    assert out["fail"].iloc[1] == "lseg div"


def test_fix_is_resumable_after_an_interrupted_run():
    """`fix()` 가 완료되기 전에 죽으면 다시 돌 수 있어야 한다 — 그 판단은 stale_px 열로 한다.

     라벨 파일이 '있다' 는 것만으로 단계를 건너뛰면, fix() 가 안 끝난 채 영영 건너뛴다."""
    import inspect
    import tools.build_nb_pricer as BNP
    src = inspect.getsource(BNP)
    assert "stale_px" in src, "노트북 1 이 fix() 완료 여부를 보지 않는다"


@pytest.mark.skipif(not (fm.CACHE / "mcvar" / "mc_variants.parquet").exists(),
                    reason="라벨 파일 없음")
def test_shipped_label_has_the_complete_schema():
    R = pd.read_parquet(fm.CACHE / "mcvar" / "mc_variants.parquet")
    missing = DONE_COLS - set(R.columns)
    assert missing == set(), "라벨이 미완성이다 (fix() 가 끝나지 않음): {}".format(sorted(missing))
    assert int(R["fail"].notna().sum()) == 0, "가격 실패가 남아 있다"
