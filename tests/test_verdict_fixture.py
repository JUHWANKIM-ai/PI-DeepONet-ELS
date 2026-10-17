# -*- coding: utf-8 -*-
"""재현 판정은 기준선 스냅샷을 지운 뒤에도 감사 가능해야 한다.

 `scratch/raw_run_baseline/` 는 git-ignore 된 되돌림용 사본이고 Task 8 에서 사람 승인 후 지운다.
 판정의 입력이 거기에만 있으면, 지우는 순간 `compare()` 가 FileNotFoundError 를 내고
 판정 테스트 대부분이 건너뛰기로 바뀌어 **결론만 남고 근거가 사라진다**.

 그래서 판정에 쓰인 세 입력을 저장소 안의 작은 픽스처로 떼어 둔다 (데이터가 아니라 **판정 근거**다).
   1. 공통난수 가격계열 측정 (source_switch_impact.csv, 200건)  -> px_mad
   2. 구본 모델 성능 ALL·통합학습 한 행                          -> model_verdict
   3. 구본 라벨 mc_B 와 상품 소속                                -> mc_mad · n_only_*
 1·2 는 작아서 그대로 담고, 3 은 부분집합별 요약통계로만 담는다 (전량은 데이터이므로 담지 않는다).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIX = Path(__file__).resolve().parent / "fixtures" / "raw_run_verdict.json"
KEYS = {"built_at", "px_effect", "model_old", "label_summary", "notes"}


def test_fixture_exists_and_is_small():
    assert FIX.exists(), "판정 근거 픽스처가 없다 — tools/freeze_verdict.py 로 만든다"
    assert FIX.stat().st_size < 200_000, "픽스처가 너무 크다 — 데이터를 담고 있다"


def test_fixture_shape():
    d = json.loads(FIX.read_text(encoding="utf-8"))
    assert set(d) == KEYS
    assert set(d["px_effect"]) == {"TIGHT", "IDX", "OTHER"}
    for s, g in d["px_effect"].items():
        assert set(g) == {"n", "mad", "bias", "se", "hi"}
    assert {"R2", "MAPE", "MAE"} <= set(d["model_old"])
    assert set(d["label_summary"]) == {"TIGHT", "IDX", "OTHER"}


def test_fixture_matches_the_live_baseline_while_it_exists():
    """스냅샷이 아직 있는 동안에는 픽스처가 그것과 일치해야 한다 — 베껴 쓴 값이 아님을 보인다."""
    from tools.verify_raw_run import BASE, px_effect
    if not (BASE / "statistics" / "source_switch_impact.csv").exists():
        pytest.skip("기준선 스냅샷이 이미 삭제됐다 — 픽스처가 유일한 근거다")
    d = json.loads(FIX.read_text(encoding="utf-8"))
    live = px_effect()
    for s, (n, mad, bias, se, hi) in live.items():
        g = d["px_effect"][s]
        assert g["n"] == n
        assert abs(g["mad"] - mad) < 1e-6
        assert abs(g["bias"] - bias) < 1e-6


def test_verdict_is_computable_without_the_baseline(monkeypatch, tmp_path):
    """스냅샷을 지운 뒤에도 판정이 돌아야 한다 — 결론만 남고 근거가 사라지면 감사 불가능하다."""
    import tools.verify_raw_run as V
    monkeypatch.setattr(V, "BASE", tmp_path / "gone")      # 삭제된 상태를 흉내낸다
    T = V.compare()
    assert list(T.subset) == ["TIGHT", "IDX", "OTHER"]
    assert T.px_mad.notna().all(), "픽스처에서 가격계열 측정을 못 읽는다"
    assert bool(T.frozen.all()), "고정 근거에서 왔다는 표시가 없다"
    Vd = V.verdict(T)
    assert Vd["TIGHT"][0] is True and Vd["IDX"][0] is None
