# -*- coding: utf-8 -*-
"""총분산 기간구조: 캘린더 차익 부재 · 기준만기 비율 1 · 보간/외삽 규칙."""
import numpy as np
import pytest

from module.iv_term import VIX_NODES, sigma_at, term_ratio, total_variance_curve

DAYS = [d for _, d in VIX_NODES]


def test_ratio_is_one_at_reference_maturity():
    row = [0.16, 0.18, 0.20, 0.22]
    assert term_ratio(row, 30 / 365) == pytest.approx(1.0)


def test_flat_term_structure_gives_ratio_one_at_any_maturity():
    row = [0.2, 0.2, 0.2, 0.2]
    for T in (0.1, 0.5, 1.0, 3.0, 5.0):
        assert term_ratio(row, T) == pytest.approx(1.0, abs=1e-12)


def test_upward_sloping_curve_raises_long_dated_vol():
    row = [0.16, 0.18, 0.20, 0.22]
    assert term_ratio(row, 3.0) > term_ratio(row, 1.0) > 1.0


def test_total_variance_is_nondecreasing_even_under_inversion():
    """역전된 곡선(위기)에서도 w 는 비감소여야 한다 — 캘린더 차익 부재."""
    t, w = total_variance_curve([0.60, 0.40, 0.25, 0.20], DAYS)
    assert np.all(np.diff(w) >= 0)


def test_interpolation_is_linear_in_total_variance_not_in_vol():
    t = np.array([0.25, 1.0]); w = np.array([0.04 * 0.25, 0.09 * 1.0])   # σ 0.20 -> 0.30
    T = 0.625
    w_lin = np.interp(T, t, w)
    assert sigma_at(t, w, T) == pytest.approx(np.sqrt(w_lin / T))
    assert sigma_at(t, w, T) != pytest.approx(0.25)                      # σ 선형보간값과 다르다


def test_extrapolation_holds_last_forward_variance_constant():
    t = np.array([93, 182]) / 365; s = np.array([0.20, 0.22])
    w = s ** 2 * t
    fwd = (w[1] - w[0]) / (t[1] - t[0])
    for T in (1.0, 3.0):
        assert sigma_at(t, w, T) ** 2 * T == pytest.approx(w[1] + fwd * (T - t[1]))


def test_missing_nodes_are_dropped_not_propagated():
    assert np.isfinite(term_ratio([np.nan, 0.18, 0.20, 0.22], 3.0))
    assert np.isnan(term_ratio([np.nan, np.nan, np.nan, 0.22], 3.0))   # 노드 1개면 곡선 불가
