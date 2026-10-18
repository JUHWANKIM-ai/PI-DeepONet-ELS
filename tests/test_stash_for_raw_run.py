# -*- coding: utf-8 -*-
"""원천 전용 실행을 위한 산출물 격리.

 `data/cache/` 전체와 `result/` 의 산출물, 그리고 `data/` 의 조립 데이터셋을 scratch 로 옮겨
 **원천(data/raw + data/share_underlyings_v2)만 남긴** 상태에서 노트북을 돌린다.

 건드리면 안 되는 것:
   data/raw/            DART 공시 3종 + rates/ 금리 커브 3파일
   data/share_underlyings_v2/   기초자산 187 CSV

 이 백업이 **유일한 되돌림 수단**이다 — Task 8 의 사람 승인 전까지 지우지 않는다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                     # noqa: E402
from generate.stash_for_raw_run import stash, KINDS        # noqa: E402


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """fm 경로를 tmp 로 돌리고 더미 산출물을 깐다."""
    data, cache, result = tmp_path / "data", tmp_path / "data" / "cache", tmp_path / "result"
    for d in (cache, result / "models", result / "statistics", result / "image",
              result / "predictions", data / "raw" / "rates", data / "share_underlyings_v2"):
        d.mkdir(parents=True, exist_ok=True)
    (cache / "px_X.parquet").write_text("x", encoding="utf-8")
    (cache / "div_X.parquet").write_text("x", encoding="utf-8")
    (result / "models" / "w.pt").write_text("x", encoding="utf-8")
    (result / "statistics" / "s.csv").write_text("x", encoding="utf-8")
    (result / "image" / "i.png").write_text("x", encoding="utf-8")
    for n in ("els3_dataset.parquet", "els12_dataset.parquet", "ml.csv", "deeponet.csv"):
        (data / n).write_text("x", encoding="utf-8")
    (data / "raw" / "rates" / "krw_curve.parquet").write_text("x", encoding="utf-8")
    (data / "share_underlyings_v2" / "master.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(fm, "DATA", data)
    monkeypatch.setattr(fm, "CACHE", cache)
    monkeypatch.setattr(fm, "RESULT", result)
    monkeypatch.setattr(fm, "RAW", data / "raw")
    return tmp_path


def test_stash_reports_every_kind(fake_tree):
    n = stash(fake_tree / "baseline")
    assert set(n) == set(KINDS)
    assert n["cache"] == 2 and n["models"] == 1 and n["datasets"] == 4


def test_stash_empties_the_sources(fake_tree):
    stash(fake_tree / "baseline")
    assert not list((fake_tree / "data" / "cache").glob("*"))
    assert not list((fake_tree / "result" / "models").glob("*"))
    assert not (fake_tree / "data" / "ml.csv").exists()


def test_stash_keeps_raw_and_underlying(fake_tree):
    """원천은 건드리지 않는다 — 이게 깨지면 재현 자체가 불가능해진다."""
    stash(fake_tree / "baseline")
    assert (fake_tree / "data" / "raw" / "rates" / "krw_curve.parquet").exists()
    assert (fake_tree / "data" / "share_underlyings_v2" / "master.csv").exists()


def test_stash_is_a_move_not_a_copy(fake_tree):
    dst = fake_tree / "baseline"
    stash(dst)
    assert (dst / "cache" / "px_X.parquet").exists()
    assert (dst / "datasets" / "ml.csv").exists()
    assert (dst / "image" / "i.png").exists()


def test_stash_refuses_existing_destination(fake_tree):
    """이미 있는 백업을 덮어쓰면 되돌림 수단을 잃는다."""
    dst = fake_tree / "baseline"
    dst.mkdir()
    (dst / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        stash(dst)
