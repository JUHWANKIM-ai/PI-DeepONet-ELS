# -*- coding: utf-8 -*-
"""공정가 시대 열(`mc` · `recent_margin`)에 대한 의존 끊기.

 `els3_dataset.parquet` 의 `mc` · `mc_seed` · `recent_margin` 은 `module.mc_shards.combine()` 이
 샤드 캐시 28개를 합쳐 붙인다. 원천 전용 재현에서는 그 캐시가 없다.

 그런데 지금 실제로 쓰이는 것은 두 가지뿐이다.
   mc_seed   MC 공통난수 — 발행일 정렬 후 행 인덱스일 뿐, 결정적 계산이다
   mc 열 존재 build_datasets 가 공통 열로 넣고 module.data.load 가 D.MC 로 읽는다
             (그 값은 module.mcvar.make_D 가 곧바로 변형 B 로 갈아끼운다)

 그래서 셋을 이렇게 가른다.
   1. mc_seed 는 build_source 가 부여한다 (MC 와 무관한 결정적 계산)
   2. build_datasets 는 mc · recent_margin 결측을 허용하되 열은 유지한다
   3. 공정가 2단계 경로(module.infer)는 mc 가 없으면 **명시적으로 거부**한다
      — 조용히 NaN 을 퍼뜨리면 안 된다
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm          # noqa: E402
import module.data as MD                      # noqa: E402
import module.infer as infer                  # noqa: E402


def test_build_datasets_tolerates_missing_anchor(tmp_path, monkeypatch):
    """mc · recent_margin 이 없는 원천으로도 ml.csv · deeponet.csv 가 만들어진다.

     열은 유지하되 NaN 으로 채운다 — module.data.load 의 스키마가 깨지면 안 되기 때문이다."""
    src = pd.read_parquet(fm.source()).head(30).copy()
    src = src.drop(columns=[c for c in ("mc", "recent_margin") if c in src.columns])
    src["issue_intensity"] = 0          # _load_source 가 붙이는 열 — 이 테스트의 관심사가 아니다
    monkeypatch.setattr(MD, "_load_source", lambda: MD.fill_lizard_neutral(src))
    monkeypatch.setattr(fm, "DATA", tmp_path)
    monkeypatch.setattr(fm, "DATASETS", {k: tmp_path / os.path.basename(str(v))
                                         for k, v in fm.DATASETS.items()})
    monkeypatch.setattr(fm, "dataset", lambda n: tmp_path / (n + ".csv"))
    info = MD.build_datasets()
    assert info["ml"] == 30
    ml = pd.read_csv(tmp_path / "ml.csv", encoding=MD.CSV_ENC)
    assert MD.ANCHOR in ml.columns and MD.MARGIN in ml.columns
    assert ml[MD.ANCHOR].isna().all()


def _fake_D_without_mc():
    n = 6
    return SimpleNamespace(
        n=n, FAIR=np.full(n, 0.95, "float32"), MC=np.full(n, np.nan, "float32"),
        rm=np.zeros(n, "float32"), ITEM=np.array([f"I{i}" for i in range(n)]),
        ORD=np.arange(n), WF=[(np.arange(4), np.arange(2, 4), np.array([4, 5]))])


def test_infer_raises_when_mc_is_missing():
    """공정가 2단계는 mc 참값이 있어야 한다 — 없으면 NaN 결과 대신 오류."""
    with pytest.raises(ValueError, match="mc"):
        infer.predict_all_from_weights(_fake_D_without_mc(), {})
