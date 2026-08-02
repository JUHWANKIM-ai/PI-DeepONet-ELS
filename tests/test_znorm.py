import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from module.data import zstats, znorm, ZCLIP


def test_constant_train_column_does_not_explode_on_unseen_test_value():
    """train 에서 상수인 열(리자드 lz_pmt_5/6/7 처럼)에 test 가 다른 값을 주면
     std=eps 로 나눠 |z| 가 2천5백만까지 튀었다. 클리핑으로 막는다."""
    a = np.zeros((10, 1), dtype="float32")
    a[9, 0] = 0.6                       # test 행에만 나타나는 값
    tr = np.arange(9)
    m, s = zstats(a, tr)
    raw = (a - m) / s
    assert abs(raw[9, 0]) > 1e6, "재현 실패: 원래는 폭주해야 한다"
    z = znorm(a, m, s)
    assert np.abs(z).max() <= ZCLIP + 1e-6


def test_near_constant_column_is_bounded():
    a = np.full((1000, 1), 1.2, dtype="float32")
    a[:2, 0] = 0.6
    tr = np.arange(900)
    m, s = zstats(a, tr)
    z = znorm(a, m, s)
    assert np.abs(z).max() <= ZCLIP + 1e-6


def test_normal_column_is_unchanged_by_clipping():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 3)).astype("float32")
    tr = np.arange(400)
    m, s = zstats(a, tr)
    raw = (a - m) / s
    z = znorm(a, m, s)
    assert np.abs(raw).max() < ZCLIP, "표본이 정규라면 클리핑에 안 걸려야 한다"
    np.testing.assert_allclose(z, raw, rtol=0, atol=1e-6)


def test_zstats_unchanged_semantics():
    a = np.array([[1.0], [2.0], [3.0], [99.0]], dtype="float32")
    m, s = zstats(a, np.arange(3))
    assert abs(float(m[0]) - 2.0) < 1e-6
    assert float(s[0]) > 0
