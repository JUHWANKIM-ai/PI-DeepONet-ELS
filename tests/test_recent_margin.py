import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from module.mc_shards import recent_margin


def _df(rows):
    return pd.DataFrame(rows, columns=["isu_ord", "fair", "mc", "opt_type", "ki_yn"])


def test_causal_uses_only_prior_issues():
    d = _df([(100, 0.90, 1.00, "STEP", 1),
             (101, 0.92, 1.00, "STEP", 1),
             (102, 0.95, 1.00, "STEP", 1)])
    rm = recent_margin(d)
    assert rm[0] == 0.0                                   # 앞선 발행 없음
    assert abs(rm[1] - (-0.10)) < 1e-6                    # 0번만
    assert abs(rm[2] - (-0.09)) < 1e-6                    # 0,1번 평균


def test_window_is_90_days():
    d = _df([(0, 0.80, 1.00, "STEP", 1),                  # 90일보다 오래됨
             (100, 0.90, 1.00, "STEP", 1),
             (150, 0.95, 1.00, "STEP", 1)])
    rm = recent_margin(d)
    assert abs(rm[2] - (-0.10)) < 1e-6                    # 100번만 (0번은 창 밖)


def test_groups_do_not_leak_into_each_other():
    """구조가 섞이면 마진이 오염된다 — 같은 구조 안에서만 평균낸다."""
    d = _df([(100, 0.70, 1.00, "LIZARD", 0),              # 마진 -0.30 (다른 구조)
             (101, 0.90, 1.00, "STEP", 1),
             (102, 0.95, 1.00, "STEP", 1)])
    rm = recent_margin(d)
    assert abs(rm[2] - (-0.10)) < 1e-6                    # LIZARD 의 -0.30 이 섞이면 안 된다
    assert rm[1] == 0.0                                   # STEP-KI 중 첫 발행


def test_returns_float32_array_aligned_to_rows():
    d = _df([(100, 0.9, 1.0, "STEP", 1), (101, 0.9, 1.0, "LIZARD", 0)])
    rm = recent_margin(d)
    assert isinstance(rm, np.ndarray) and rm.dtype == np.float32 and len(rm) == 2


def test_unsorted_input_is_handled():
    d = _df([(102, 0.95, 1.00, "STEP", 1),
             (100, 0.90, 1.00, "STEP", 1),
             (101, 0.92, 1.00, "STEP", 1)])
    rm = recent_margin(d)
    assert abs(rm[0] - (-0.09)) < 1e-6                    # 102 는 100,101 평균
    assert rm[1] == 0.0
