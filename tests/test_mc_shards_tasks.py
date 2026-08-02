import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from module import mc_shards as S


@pytest.fixture(scope="module")
def dt():
    return S._tasks()


def test_tasks_are_dicts_with_schedule_arrays(dt):
    df, tasks = dt
    assert len(tasks) == len(df)
    t = tasks[0]
    for k in ("i", "item", "iord", "B", "c", "ten", "sig_eff",
              "strikes", "pmts", "lz_barr", "lz_pmt"):
        assert k in t, f"missing key {k}"
    n = len(t["strikes"])
    assert n >= 2
    assert len(t["pmts"]) == n and len(t["lz_barr"]) == n and len(t["lz_pmt"]) == n
    assert int(df.loc[t["i"], "nobs"]) == n           # 실제 회차 수만큼만 잘라 넘긴다


def test_no_ki_products_have_barrier_one(dt):
    df, tasks = dt
    noki = [t for t in tasks if int(df.loc[t["i"], "ki_yn"]) == 0]
    assert noki, "노낙인 상품이 없다 (Task 2 재빌드 필요)"
    assert all(t["B"] == 1.0 for t in noki[:50])


def test_step_products_have_no_lizard(dt):
    df, tasks = dt
    step = [t for t in tasks if str(df.loc[t["i"], "opt_type"]) == "STEP"]
    assert step, "STEP 상품이 하나도 없다"
    assert all(not np.isfinite(np.asarray(t["lz_barr"], float)).any() for t in step[:50])


def test_lizard_products_have_at_least_one_barrier(dt):
    df, tasks = dt
    lz = [t for t in tasks if str(df.loc[t["i"], "opt_type"]) == "LIZARD"]
    assert lz, "LIZARD 상품이 하나도 없다"
    assert all(np.isfinite(np.asarray(t["lz_barr"], float)).any() for t in lz[:50])
