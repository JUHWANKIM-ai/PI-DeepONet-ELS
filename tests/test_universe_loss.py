# -*- coding: utf-8 -*-
"""유니버스에서 빠지는 건수를 **측정해 고정**한다 — 사유를 모르는 것과 크기를 모르는 것은 다르다.

`udly_ticker_map.json` 의 52개 항목이 `None` 이고, 그 52개 **전부 master.csv 에 가격 행이 있다.**
그 때문에 3자산 공시 **1,020건**이 데이터셋에 못 들어온다 — `^SX7E` 수정으로 복구한 124건의
8배다.

커버리지 테스트는 이 채널을 볼 수 없었다. `module/mc.py` 도 테스트도 빈 티커를 **대체 경로가
작동하기 전에** 건너뛰기 때문이다(`if not t: continue`). 그래서 "티커가 해석되는가" 는
통과하면서 "왜 1,020건이 없는가" 는 아무도 묻지 않았다.

**이 테스트는 1,020건이 들어와야 한다고 주장하지 않는다.** 목록에는 원자재(BRENT·GOLD·SILVER)와
디크리먼트/NTR/레버리지 변형이 섞여 있어 배당 처리가 다르거나 의도적으로 범위 밖일 수 있다.
주장하는 것은 하나다 — **그 크기가 상수로 기록돼 있어야 하고, 지도가 바뀌면 이 수가 움직여야 한다.**
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm  # noqa: E402

# 2026-10-18 실측. 지도를 고치면 이 수가 바뀌고 테스트가 그 사실을 알린다.
N_NONE = 52
N_NONE_WITH_PRICE = 52
N_FILINGS_LOST = 1020


def _none_ids():
    m = json.loads((fm.RAW / "udly_ticker_map.json").read_text(encoding="utf-8"))
    return {k for k, v in m.items() if not v}, len(m)


def test_none_mapped_count_is_pinned():
    ids, total = _none_ids()
    assert (len(ids), total) == (N_NONE, 100), "지도가 바뀌었다: None {} / {}".format(
        len(ids), total
    )


def test_none_mapped_underlyings_have_price_data_on_disk():
    """사유가 '자료 없음' 이 아니라는 것 — 전부 가격 행이 있다. 즉 **선별**이다."""
    ids, _ = _none_ids()
    M = pd.read_csv(fm.DATA / "share_underlyings_v2" / "master.csv", encoding="utf-8-sig")
    have = set(M.UDLY_ID.astype(str)) & ids
    assert len(have) == N_NONE_WITH_PRICE, "{} / {}".format(len(have), len(ids))


def test_filings_lost_to_none_mappings_is_pinned():
    """커버리지 테스트가 못 보는 채널의 크기. ^SX7E 복구 124건과 나란히 둘 수 있어야 한다."""
    ids, _ = _none_ids()
    ud = pd.read_csv(fm.RAW / "LAKE_V2_DART_UDLY_INFO.csv", usecols=["ITEM_CD", "UDLY_ID"])
    g = ud.groupby("ITEM_CD").UDLY_ID.apply(set)
    three = g[g.apply(len) == 3]
    lost = three[three.apply(lambda s: bool(s & ids))]
    assert len(lost) == N_FILINGS_LOST, "None 매핑 손실이 {}건으로 바뀌었다 (기록 {})".format(
        len(lost), N_FILINGS_LOST
    )


def test_recovered_count_is_recorded_for_comparison():
    """복구분도 같이 — 1,020 대 124 라는 비율이 이 결정의 맥락이다."""
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet", columns=["udl1", "udl2", "udl3"])
    assert int((d == "^SX7E").any(axis=1).sum()) == 124
