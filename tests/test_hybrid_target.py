import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
import numpy as np
import pandas as pd


def _fake_D(n=6):
    return SimpleNamespace(
        n=n,
        WF=[(np.array([0, 1, 2]), np.array([3]), np.array([4, 5]))],
        FAIR=np.array([0.90, 0.95, 0.85, 0.92, 1.00, 0.80], dtype="float32"),
        MC=np.array([0.98, 0.99, 0.95, 0.97, 1.03, 0.93], dtype="float32"),
        rm=np.array([-0.05, -0.04, -0.06, -0.05, -0.02, -0.07], dtype="float32"),
        ITEM=np.array([f"IT{i}" for i in range(n)]),
        ORD=np.arange(n, dtype=float),
    )


def _empty():
    return pd.DataFrame({"ITEM_CD": [], "y_true": [], "y_pred": []})


def _capture(monkeypatch, module):
    """해당 모듈의 predict_hybrid 를 가로채 호출 인자를 수집한다."""
    got = []

    def fake(D, cfg, anchor_fn, resid_fn=None, name=None, target=None, use_margin=True):
        got.append(dict(name=name, target=target, use_margin=use_margin))
        return _empty()

    monkeypatch.setattr(module, "predict_hybrid", fake)
    return got


def test_deeponet_hybrids_target_fair_with_margin(monkeypatch):
    import model.deeponet as M
    got = _capture(monkeypatch, M)
    D = _fake_D()
    M.run(D, {})
    assert got, "predict_hybrid 가 호출되지 않았다"
    for c in got:
        assert c["use_margin"] is True, f"{c['name']}: use_margin={c['use_margin']}"
        np.testing.assert_array_equal(c["target"], D.FAIR)


def test_xgb_hybrid_targets_fair_with_margin(monkeypatch):
    import model.benchmark as B
    got = _capture(monkeypatch, B)
    monkeypatch.setattr(B, "predict_direct_tab", lambda *a, **k: _empty())
    D = _fake_D()
    B.run(D, {})
    assert got, "predict_hybrid 가 호출되지 않았다"
    for c in got:
        assert c["use_margin"] is True
        np.testing.assert_array_equal(c["target"], D.FAIR)


def test_direct_benchmarks_target_fair(monkeypatch):
    import model.benchmark as B
    got = []

    def fake_direct(D, cfg, key, name=None, target=None):
        got.append(dict(name=name, target=target))
        return _empty()

    monkeypatch.setattr(B, "predict_direct_tab", fake_direct)
    monkeypatch.setattr(B, "predict_hybrid", lambda *a, **k: _empty())
    D = _fake_D()
    B.run(D, {})
    assert got, "predict_direct_tab 이 호출되지 않았다"
    for c in got:
        np.testing.assert_array_equal(c["target"], D.FAIR)


def test_infer_assembles_fair_with_margin(monkeypatch):
    """저장 가중치 재조립 경로도 공정가 규약을 따라야 한다."""
    from module import infer
    D = _fake_D()
    mc_hat = np.array([0.90, 0.91, 0.92, 0.93, 0.94, 0.95], dtype="float32")
    resid = np.array([0.001, 0.002], dtype="float32")
    monkeypatch.setattr(infer, "_DIRECT", {})
    monkeypatch.setattr(infer, "_HYBRID", {"fake": ("curve", "xgb")})
    monkeypatch.setattr(infer, "_anchor_loader", lambda D, a, b: (lambda idx: mc_hat[np.asarray(idx)]))
    monkeypatch.setattr(infer, "_resid_loader", lambda D, r, b: (lambda idx: resid))
    out = infer.predict_all_from_weights(D, {})["fake"]
    te = np.array([4, 5])
    np.testing.assert_allclose(out["y_true"].values, D.FAIR[te], atol=1e-6)
    np.testing.assert_allclose(out["y_pred"].values, mc_hat[te] + D.rm[te] + resid, atol=1e-6)
    np.testing.assert_allclose(out["resid_true"].values,
                               D.FAIR[te] - mc_hat[te] - D.rm[te], atol=1e-6)
    np.testing.assert_allclose(out["mc_true"].values, D.MC[te], atol=1e-6)
    np.testing.assert_allclose(out["mc_pred"].values, mc_hat[te], atol=1e-6)
