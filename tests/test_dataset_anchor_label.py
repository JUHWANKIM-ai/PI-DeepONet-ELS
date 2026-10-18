# -*- coding: utf-8 -*-
"""`ml.csv`·`deeponet.csv` 의 `MC` 열은 정본 라벨이어야 한다.

Task 4 에서 `mc` 열 의존을 끊어 **열만 만들고 NaN 으로** 두게 했다. 학습은 `module.mcvar` 가
라벨을 직접 공급하므로 그것으로 충분했지만, README 가 데이터셋으로 내세우는 두 CSV 의
라벨 열이 58,913행 전부 비게 됐다. 빈 라벨 열은 조용히 오용된다.

라벨 파일(`mcvar/mc_variants.parquet`)이 있으면 그 `mc_B` 를 채우고, 없으면 NaN 으로 둔다
(차가운 트리의 첫 노트북 0 실행 시점에는 아직 라벨이 없다 — 노트북 1 이 만든 뒤 다시 채운다).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm  # noqa: E402
import module.data as DA  # noqa: E402


def test_fill_anchor_uses_the_canonical_label(tmp_path, monkeypatch):
    lab = tmp_path / "mc_variants.parquet"
    pd.DataFrame({"item": ["A", "B"], "mc_B": [0.97, 1.01]}).to_parquet(lab)
    monkeypatch.setattr(DA, "_label_path", lambda: lab)
    df = pd.DataFrame({"item": ["A", "B", "C"], "mc": [np.nan] * 3})
    n = DA.fill_anchor(df)
    assert n == 2
    assert df.loc[0, "mc"] == pytest.approx(0.97)
    assert np.isnan(df.loc[2, "mc"])  # 라벨 없는 행은 NaN 으로 남는다


def test_fill_anchor_is_a_noop_without_a_label_file(tmp_path, monkeypatch):
    """차가운 트리에서 노트북 0 이 먼저 돈다 — 그때는 조용히 넘어가야 한다."""
    monkeypatch.setattr(DA, "_label_path", lambda: tmp_path / "nope.parquet")
    df = pd.DataFrame({"item": ["A"], "mc": [np.nan]})
    assert DA.fill_anchor(df) == 0
    assert np.isnan(df.loc[0, "mc"])


def test_fill_anchor_does_not_overwrite_existing_values(tmp_path, monkeypatch):
    """이미 값이 있으면 건드리지 않는다 — 공정가 시대 `mc` 를 덮어쓰면 안 된다."""
    lab = tmp_path / "mc_variants.parquet"
    pd.DataFrame({"item": ["A"], "mc_B": [0.5]}).to_parquet(lab)
    monkeypatch.setattr(DA, "_label_path", lambda: lab)
    df = pd.DataFrame({"item": ["A"], "mc": [0.99]})
    assert DA.fill_anchor(df) == 0
    assert df.loc[0, "mc"] == pytest.approx(0.99)


@pytest.mark.skipif(not fm.dataset("ml").exists(), reason="ml.csv 없음")
def test_shipped_ml_csv_carries_the_label():
    """전 과정(0->3)을 돌린 뒤에는 라벨이 비어 있을 수 없다."""
    if not (fm.CACHE / "mcvar" / "mc_variants.parquet").exists():
        pytest.skip("라벨 파일이 없다 (노트북 1 미실행)")
    d = pd.read_csv(fm.dataset("ml"), usecols=["MC"], encoding=DA.CSV_ENC)
    assert d.MC.notna().mean() > 0.99, "ml.csv 의 MC 열이 비었다"
