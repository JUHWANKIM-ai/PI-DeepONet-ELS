import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from util.metric import metrics_by


def test_groups_are_split_and_counted():
    y = np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2])
    p = np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    out = metrics_by(y, p, g, min_n=1)
    assert set(out.index) == {"a", "b"}
    assert out.loc["a", "n"] == 3 and out.loc["b", "n"] == 3
    assert out.loc["a", "R2"] > 0.99


def test_small_groups_are_merged():
    y = np.arange(10, dtype=float) + 1.0
    p = y + 0.01
    g = np.array(["big"] * 8 + ["tiny1", "tiny2"])
    out = metrics_by(y, p, g, min_n=3)
    assert set(out.index) == {"big", "(small)"}
    assert out.loc["(small)", "n"] == 2


def test_returns_all_metric_columns():
    y = np.array([1.0, 1.1, 1.2, 1.3])
    p = np.array([1.05, 1.08, 1.25, 1.28])
    out = metrics_by(y, p, np.array(["x"] * 4), min_n=1)
    for c in ("n", "R2", "MAE", "RMSE", "MAPE%", "MdAPE%", "Bias%", "Spearman"):
        assert c in out.columns


def test_sorted_by_sample_size_desc():
    y = np.arange(9, dtype=float) + 1.0
    p = y + 0.01
    g = np.array(["a"] * 2 + ["b"] * 3 + ["c"] * 4)
    out = metrics_by(y, p, g, min_n=1)
    assert list(out.index) == ["c", "b", "a"]


def test_accepts_pandas_series():
    y = pd.Series([1.0, 1.1, 1.2, 2.0, 2.1, 2.2], index=[10, 11, 12, 13, 14, 15])
    p = pd.Series([1.01, 1.09, 1.22, 2.02, 2.08, 2.25], index=y.index)
    g = pd.Series(["a", "a", "a", "b", "b", "b"], index=y.index)
    out = metrics_by(y, p, g, min_n=1)
    assert out.loc["a", "n"] == 3 and out.loc["b", "n"] == 3
