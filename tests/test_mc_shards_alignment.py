# -*- coding: utf-8 -*-
"""샤드 캐시 병합은 **정렬을 증명할 수 없으면 거부**해야 한다.

`combine()` 은 샤드 JSON 의 키를 **행 번호**로 보고 현재 데이터셋에 꽂는다. 그런데 이 브랜치에서
유니버스가 58,790 -> 59,037 로 늘었고 새 상품 124건이 발행일 정렬 중간중간에 끼어들었다.
그러면 첫 삽입 지점 이후의 모든 샤드 결과가 **다른 상품에** 꽂힌다. 행 수·상품 동일성 검사가
없어 조용히 어긋난다.

게다가 샤드는 **행 번호 시드**로 가격을 매겼는데, combine 이 내용 기반 시드를 덮어써
그 값들이 갖지 않은 출처를 주장하고 있었다 (이 브랜치에서 내가 넣은 결함이다).

샤드 JSON 에는 상품코드가 없어 정렬을 직접 확인할 방법이 없다. 그래서 할 수 있는 유일한
검사가 **행 수 일치**이고, 어긋나면 거부한다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm  # noqa: E402
import module.mc_shards as MS  # noqa: E402
import module.mc as MC  # noqa: E402


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """행 3개짜리 데이터셋 + 그에 맞는 샤드 1개."""
    df = pd.DataFrame({"item": ["A", "B", "C"], "fair": [0.95, 0.96, 0.97]})
    monkeypatch.setattr(fm, "CACHE", tmp_path)
    monkeypatch.setattr(MS, "_tasks", lambda: (df.copy(), None))
    monkeypatch.setattr(MS, "_save", lambda d: d) if hasattr(MS, "_save") else None

    def write_shard(n_rows):
        payload = {str(i): [0.9] * len(MC.MC_COLS) for i in range(n_rows)}
        (tmp_path / "mccal_shard_0.json").write_text(json.dumps(payload), encoding="utf-8")

    return df, write_shard


def test_combine_refuses_when_the_shard_row_count_differs(fake, monkeypatch):
    """유니버스가 바뀐 뒤의 샤드는 행 번호가 어긋난다 — 조용히 꽂으면 안 된다."""
    df, write_shard = fake
    write_shard(5)  # 데이터셋은 3행인데 샤드는 5행 기준
    with pytest.raises(ValueError, match="행 수"):
        MS.combine(N=1, verbose=False)


def test_combine_does_not_claim_a_seed_the_values_lack(fake, monkeypatch):
    """샤드는 행 번호 시드로 매겨졌다 — 내용 기반 시드를 덮어쓰면 거짓 출처가 된다."""
    import inspect

    src = inspect.getsource(MS.combine)
    assert "mc_seed_for" not in src, "combine 이 가지지 않은 출처를 주장한다"
