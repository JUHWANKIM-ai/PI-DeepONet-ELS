# -*- coding: utf-8 -*-
"""공통난수 시드는 **상품 내용**에서 나와야 한다 — 행 번호면 유니버스가 바뀔 때 전부 재배치된다.

 종전에는 `mc_seed = np.arange(len(df))`, 즉 발행일 정렬 후 **행 번호**였다. 그래서 상품 123건이
 새로 들어오자 공통 상품의 **98.2%(57,705/58,790)** 가 다른 난수를 받았고, 그 결과

   - 구본과 신본이 공통난수가 아니게 되어 **라벨 차이로는 재현을 판정할 수 없게** 됐다
   - 판정을 200건짜리 별도 측정(px_mad, TIGHT 는 n=11)으로 우회해야 했다

 `item` 의 안정적 해시를 쓰면 유니버스가 바뀌어도 시드가 고정되고, 앞으로 모든 라벨 실행이
 비트 단위로 비교 가능해진다. Python 내장 `hash()` 는 프로세스마다 소금이 달라 쓸 수 없다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm          # noqa: E402
from module.seeds import mc_seed_for, SEED_MAX   # noqa: E402

ITEMS = ["KR6523305516", "KR6513308512", "KR6523313510"]


def test_seed_is_a_function_of_the_item_only():
    a = mc_seed_for(ITEMS)
    b = mc_seed_for(list(reversed(ITEMS)))[::-1]
    assert (a == b).all(), "순서가 바뀌면 시드가 달라진다 — 행 번호 의존이 남아 있다"


def test_inserting_products_does_not_move_other_seeds():
    """이 성질이 전부다 — 유니버스가 늘어도 기존 상품의 난수가 그대로여야 대조가 가능하다."""
    before = mc_seed_for(ITEMS)
    grown = ITEMS[:1] + ["KR9NEWPRODUCT"] + ITEMS[1:]
    after = mc_seed_for(grown)
    keep = [i for i, it in enumerate(grown) if it in ITEMS]
    assert (after[keep] == before).all()


def test_seed_is_stable_across_processes():
    """내장 hash() 는 프로세스마다 소금이 달라 쓸 수 없다 — 값을 못박아 둔다."""
    assert int(mc_seed_for(["KR6523305516"])[0]) == int(mc_seed_for(["KR6523305516"])[0])
    import subprocess
    out = subprocess.run([sys.executable, "-c",
                          "import sys; sys.path.insert(0, r'{}'); "
                          "from module.seeds import mc_seed_for; "
                          "print(int(mc_seed_for(['KR6523305516'])[0]))".format(
                              os.path.dirname(os.path.dirname(os.path.abspath(__file__))))],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == int(mc_seed_for(["KR6523305516"])[0])


def test_dtype_and_range():
    s = mc_seed_for(ITEMS)
    assert s.dtype == np.int32
    assert (s >= 0).all() and (s < SEED_MAX).all()


@pytest.mark.skipif(not (fm.DATA / "els3_dataset.parquet").exists(), reason="데이터셋 없음")
def test_collision_rate_is_acceptable_on_the_real_universe():
    """충돌은 두 상품이 같은 난수열을 쓴다는 뜻일 뿐 편향을 만들지 않지만, 수는 알고 있어야 한다."""
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet", columns=["item"])
    s = mc_seed_for(d.item)
    dup = len(s) - len(set(s.tolist()))
    assert dup < len(s) * 0.001, "충돌 {}건 / {}".format(dup, len(s))


def test_every_assignment_site_uses_the_shared_helper():
    """네 곳에 복사돼 있던 것을 한 곳으로 모았다 — 되돌아가면 유니버스 변경에 또 깨진다."""
    import inspect
    import importlib
    # module.mc_shards 는 **일부러 뺀다.** 그 경로의 mc 값은 행 번호 시드로 계산된
    # 공정가 시대 샤드에서 오므로, 내용 기반 시드를 적으면 값이 갖지 않은 출처를 주장하게 된다.
    # 거기서는 실제로 쓰인 시드를 그대로 기록하는 것이 맞다 (tests/test_mc_shards_alignment.py).
    for name in ("module.build_source",
                 "tools.build_mc_variants", "tools.mc_check_12star"):
        src = inspect.getsource(importlib.import_module(name))
        assert "mc_seed_for" in src, "{} 가 공용 헬퍼를 쓰지 않는다".format(name)
        assert 'mc_seed"] = np.arange' not in src, "{} 에 행 번호 시드가 남아 있다".format(name)


def test_seeds_stale_detects_a_changed_seed_definition():
    """저장된 데이터셋이 지금의 시드 정의와 다르면 낡은 것이다.

     노트북 0 은 `REBUILD=False` 면 저장본을 그대로 읽었다. 그래서 시드 정의를 바꿔도
     데이터셋이 갱신되지 않았고, 행 번호 시드가 그대로 남아 라벨 재생성이 헛돌았다 —
     라벨 스탬프에서 고친 것과 같은 결함("파일이 있는가" vs "낡았는가")이 여기 또 있었다."""
    from module.seeds import seeds_stale
    d = pd.DataFrame({"item": ITEMS})
    d["mc_seed"] = mc_seed_for(d.item)
    assert seeds_stale(d) is False
    d["mc_seed"] = np.arange(len(d), dtype="int32")      # 행 번호로 되돌린 상태
    assert seeds_stale(d) is True
    assert seeds_stale(pd.DataFrame({"item": ITEMS})) is True      # 열이 아예 없다


def test_notebook0_keys_rebuild_on_seed_staleness():
    import inspect
    import tools.build_nb_data as BND
    src = inspect.getsource(BND)
    assert "seeds_stale" in src, "노트북 0 이 저장본의 시드를 검사하지 않는다"
    assert "REBUILD = False" not in src, "REBUILD 가 상수로 고정돼 있다"
