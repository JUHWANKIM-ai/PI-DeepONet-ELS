# -*- coding: utf-8 -*-
"""낡음 판단은 "파일이 있는가" 가 아니라 "무엇으로 만든 파일인가" 로 해야 한다.

 라벨이 **바뀌면** 데이터셋의 MC 열과 저장된 가중치가 둘 다 낡는다. 그런데 종전 조건은
   - 데이터셋 갱신: `ml.csv` 의 MC 가 **전부 NaN** 일 때만  -> 바뀐 경우는 안 걸린다
   - 재학습: `RETRAIN` 기본 0 + 파일 존재 여부          -> 낡은 가중치가 재사용된다
 였다. 둘 다 라벨 출처 스탬프와 비교하도록 바꾼다.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                  # noqa: E402
import module.data as DA                              # noqa: E402
import tools.build_mc_variants as BMV                 # noqa: E402


def test_dataset_stamp_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "DATA", tmp_path)
    assert DA.dataset_stamp() is None                 # 기록이 없으면 None
    DA.record_dataset_stamp({"npath": 7, "div": "lseg"})
    assert DA.dataset_stamp()["npath"] == 7


def test_datasets_are_stale_when_the_label_stamp_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "DATA", tmp_path)
    lab = tmp_path / "lab.parquet"
    BMV.write_labeled(pd.DataFrame({"item": ["A"], "mc_B": [0.9]}), lab)
    monkeypatch.setattr(DA, "_label_path", lambda: lab)
    assert DA.datasets_stale() is True                # 아직 기록이 없다
    DA.record_dataset_stamp(BMV.read_stamp(lab))
    assert DA.datasets_stale() is False
    BMV.write_labeled(pd.DataFrame({"item": ["A", "B"], "mc_B": [0.9, 1.0]}), lab)
    assert DA.datasets_stale() is True                # 유니버스가 바뀌었다


def test_build_datasets_records_the_stamp_it_filled_from(tmp_path, monkeypatch):
    """채운 라벨의 출처를 남겨야 다음 실행이 '바뀌었는가' 를 물을 수 있다."""
    src = pd.read_parquet(fm.source()).head(10).copy()
    src = src.drop(columns=[c for c in ("mc", "recent_margin") if c in src.columns])
    src["issue_intensity"] = 0
    monkeypatch.setattr(DA, "_load_source", lambda: DA.fill_lizard_neutral(src))
    monkeypatch.setattr(fm, "DATA", tmp_path)
    monkeypatch.setattr(fm, "dataset", lambda n: tmp_path / (n + ".csv"))
    monkeypatch.setattr(fm, "DATASETS", {k: tmp_path / os.path.basename(str(v))
                                         for k, v in fm.DATASETS.items()})
    DA.build_datasets()
    assert DA.dataset_stamp() is not None


def test_notebook2_keys_retrain_on_the_label_stamp():
    """노트북 2 가 스탬프로 재학습을 판단해야 한다 — 파일 존재만 보면 낡은 가중치를 쓴다."""
    import inspect
    import tools.build_nb_training as BNT
    src = inspect.getsource(BNT)
    assert "read_stamp" in src or "weights_stale" in src, "노트북 2 가 스탬프를 보지 않는다"


def test_notebook1_refreshes_datasets_on_stamp_change():
    import inspect
    import tools.build_nb_pricer as BNP
    src = inspect.getsource(BNP)
    assert "datasets_stale" in src, "노트북 1 이 '전부 NaN' 만 보고 있다"
    assert 'MC.isna().all()' not in src
