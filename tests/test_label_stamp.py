# -*- coding: utf-8 -*-
"""라벨 파일은 자기 출처를 지니고 다녀야 한다.

 `mc_variants.parquet` 는 이름이 그대로인 채 **의미가 바뀌었다** — 종전에는 yf 이산배당·40,000
 경로·58,790건이었고 지금은 LSEG 연속 q·100,000 경로·58,913건이다. 파일 안에 그 사실이 없으면

   (1) 되돌림이 깨진다 — 보관본의 yf 라벨도 `mc_variants.parquet` 라는 같은 이름이다.
   (2) 데이터셋 갱신이 안 돈다 — 노트북 1 의 조건이 "MC 가 전부 NaN" 이라서, 라벨이 **바뀐**
       경우는 걸리지 않고 낡은 라벨이 조용히 배포된다.
   (3) 재학습이 안 돈다 — RETRAIN 기본값이 0 이라 낡은 라벨로 학습된 가중치가 재사용된다.

 세 문제 모두 "파일이 있는가" 가 아니라 "무엇으로 만든 파일인가" 를 물으면 사라진다.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                  # noqa: E402
import tools.build_mc_variants as BMV                # noqa: E402

KEYS = {"npath", "div", "px_source", "n_rows", "built_at"}


def test_stamp_keys_are_pinned():
    assert set(BMV.stamp()) == KEYS
    st = BMV.stamp(n_rows=7)
    assert st["npath"] == BMV.NPATH and st["div"] == BMV.DIV and st["n_rows"] == 7


def test_stamp_round_trips_through_parquet(tmp_path):
    """parquet 메타데이터로 실어 보내고 되읽는다 — 열을 늘리면 기존 소비자가 깨진다."""
    f = tmp_path / "lab.parquet"
    d = pd.DataFrame({"item": ["A"], "mc_B": [0.9]})
    BMV.write_labeled(d, f)
    got = BMV.read_stamp(f)
    assert set(got) == KEYS
    assert int(got["npath"]) == BMV.NPATH
    assert list(pd.read_parquet(f).columns) == ["item", "mc_B"]     # 열은 그대로


def test_read_stamp_is_none_for_an_unstamped_file(tmp_path):
    f = tmp_path / "plain.parquet"
    pd.DataFrame({"item": ["A"]}).to_parquet(f)
    assert BMV.read_stamp(f) is None


def test_stamp_differs_is_the_single_comparator(tmp_path):
    """`stale()` 은 테스트만 부르던 죽은 코드라 지웠다. 비교는 stamp_differs 한 곳에서만 한다.

     판정 대상은 tests/test_stamp_unified.py 가 따로 못박는다 — 여기서는 라벨 파일에서
     읽은 스탬프가 그 비교에 그대로 쓰인다는 것만 확인한다."""
    f = tmp_path / "lab.parquet"
    BMV.write_labeled(pd.DataFrame({"item": ["A"], "mc_B": [0.9]}), f)
    st = BMV.read_stamp(f)
    assert BMV.stamp_differs(st, dict(st)) is False
    assert BMV.stamp_differs(st, dict(st, n_rows=2)) is True          # 유니버스가 바뀌었다
    assert BMV.stamp_differs(BMV.read_stamp(tmp_path / "nope.parquet"), st) is True


@pytest.mark.skipif(not (fm.CACHE / "mcvar" / "mc_variants.parquet").exists(),
                    reason="라벨 파일 없음 (노트북 1 미실행)")
def test_shipped_label_carries_a_stamp():
    """배포된 라벨에 스탬프가 있어야 한다 — 없으면 무엇으로 만든 것인지 알 수 없다."""
    got = BMV.read_stamp(fm.CACHE / "mcvar" / "mc_variants.parquet")
    assert got is not None, "라벨 파일에 출처 스탬프가 없다"
    assert int(got["npath"]) == 100_000
    assert got["div"] == "lseg"
    assert int(got["n_rows"]) == len(pd.read_parquet(fm.DATA / "els3_dataset.parquet",
                                                     columns=["item"]))
