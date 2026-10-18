# -*- coding: utf-8 -*-
"""스탬프 비교는 **한 함수**에서만 하고, 증명할 수 없으면 낡은 것으로 본다.

"통일했다" 고 원장에 적었지만 실제로는 같은 비교식이 네 벌이 됐다 —
`stale` · `step_stale` · `datasets_stale` · `weights_stale`. 키 튜플이 네 번 복사돼
`stamp()` 에 키를 하나 더하면 세 곳이 그걸 무시한다.

더 나쁜 것은 **스탬프가 없으면 전부 "최신"으로 보고**한다는 점이다. 그러면 스탬프 이전
라벨이 놓인 트리에서 네 단계가 전부 건너뛰어지고, 데이터셋도 가중치도 갱신되지 않는다 —
이 장치가 막으려던 바로 그 실패다. 가중치에 대해서는 "증명할 수 없으면 보수적으로 다시"를
택했으면서 같은 질문에 반대로 답한 셈이다.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm  # noqa: E402
import generate.build_mc_variants as BMV  # noqa: E402


def test_one_comparator_only():
    """비교식이 또 복사되면 키를 추가할 때 일부가 무시한다."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    hits = []
    for pat in ("module/*.py", "generate/*.py"):
        for f in root.glob(pat):
            txt = f.read_text(encoding="utf-8", errors="replace")
            if 'for k in ("npath"' in txt or '"n_rows")' in txt and "any(str(got" in txt:
                hits.append(f.name)
    assert len(hits) <= 1, "스탬프 비교식이 여러 곳에 있다: {}".format(sorted(hits))
    assert hasattr(BMV, "stamp_differs"), "공용 비교 함수가 없다"
    assert hasattr(BMV, "STALE_KEYS")


def test_unstamped_label_is_treated_as_stale():
    """증명할 수 없으면 낡은 것으로 본다 — 가중치에 적용한 기준과 같아야 한다."""
    assert BMV.stamp_differs(None, {"npath": 1}) is True
    assert BMV.stamp_differs({"npath": 1}, None) is True


def test_identical_stamps_are_not_stale():
    st = {k: "x" for k in BMV.STALE_KEYS}
    assert BMV.stamp_differs(dict(st), dict(st)) is False
    other = dict(st)
    other[BMV.STALE_KEYS[0]] = "y"
    assert BMV.stamp_differs(other, st) is True


def test_dead_stale_helper_is_gone_or_used():
    """`stale()` 은 테스트만 부르던 죽은 코드였다."""
    import inspect

    src = inspect.getsource(BMV)
    if "def stale(" in src:
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        callers = [
            f.name
            for pat in ("module/*.py", "generate/*.py")
            for f in root.glob(pat)
            if "BMV.stale(" in f.read_text(encoding="utf-8", errors="replace")
        ]
        assert callers, "stale() 이 아무 데서도 쓰이지 않는다 (죽은 코드)"
