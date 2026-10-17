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
    for name in ("module.build_source", "module.mc_shards",
                 "tools.build_mc_variants", "tools.mc_check_12star"):
        src = inspect.getsource(importlib.import_module(name))
        assert "mc_seed_for" in src, "{} 가 공용 헬퍼를 쓰지 않는다".format(name)
        assert 'mc_seed"] = np.arange' not in src, "{} 에 행 번호 시드가 남아 있다".format(name)
