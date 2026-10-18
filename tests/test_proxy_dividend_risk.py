# -*- coding: utf-8 -*-
"""ETF 프록시 배당은 이 파이프라인에서 가장 검증이 약한 입력이다 — 크기를 기록해 둔다.

`^SX7E`(EURO STOXX Banks)는 지수 자체의 배당수익률이 0행이라 프록시가 불가피하다.
그런데 그 프록시(`EXX1.DE`)는 **상한(30-15) NR 변형**을 추종하고, 대상은 상한 없는 가격지수다.
근사가 세 겹이다: ETF 분배(원천징수 후) <- NR 벤치마크 <- 30-15 상한 변형.

그리고 실제로 주입되는 q 가 데이터셋 전체에서 가장 크다 (실측).
  ^SX7E : 중위 6.82%  최대 9.12%
  전체   : 중위 2.06%  p99 4.89%  최대 9.12%  (최대값이 ^SX7E 다)

민감도도 실측했다 — 30건 표본에서 q 를 +-20% 흔들면 가격 스프레드가 평균 76.5원으로,
기준 모형 MAE(37.1원)의 두 배다. 즉 이 124건의 가격은 가장 덜 검증된 입력에 좌우된다.

원장 142 는 이 위험을 "중위 4.16%, 타당" 으로 적었는데, 그건 ETF 수익률 계열의 중위값이지
**실제 주입된 q** 가 아니었다. 그 오판을 수치로 못박는다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm  # noqa: E402
from module import dividend as DV  # noqa: E402

PROXY_ONLY = {"000300.SS", "^SX7E"}  # 지수 자체 수익률이 없어 ETF 로만 가능한 것
Q_SPREAD_KRW = 76.5  # +-20% 흔들었을 때 가격 스프레드 (30건 표본 실측)


def test_proxy_only_underlyings_are_declared():
    """프록시로만 배당이 가능한 기초자산을 목록으로 못박는다 — 새로 생기면 검토 대상이다."""
    for t in PROXY_ONLY:
        assert t in DV.PROXY, "{} 의 프록시가 사라졌다".format(t)
        assert DV.lseg_yield(t) is None, "{} 가 이제 자체 수익률을 가진다 — 프록시 불필요".format(t)


@pytest.mark.skipif(not (fm.CACHE / "mcvar" / "mc_variants.parquet").exists(), reason="라벨 없음")
def test_sx7e_carries_the_largest_injected_dividend():
    """이 사실을 모르고 넘어가면 안 된다 — 124건이 전체 최대 q 위에 서 있다."""
    V = pd.read_parquet(
        fm.CACHE / "mcvar" / "mc_variants.parquet", columns=["item", "q_1", "q_2", "q_3"]
    )
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet", columns=["item", "udl1", "udl2", "udl3"])
    J = V.merge(d, on="item")
    qs = [
        r["q_%d" % k]
        for _, r in J.iterrows()
        for k, u in enumerate(["udl1", "udl2", "udl3"], 1)
        if r[u] == "^SX7E"
    ]
    if not qs:
        pytest.skip("^SX7E 상품이 없다")
    allq = J[["q_1", "q_2", "q_3"]].to_numpy().ravel()
    allq = allq[np.isfinite(allq)]
    assert np.median(qs) > 3 * np.median(allq), "^SX7E 의 q 가 더 이상 두드러지지 않는다"
    assert abs(max(qs) - allq.max()) < 1e-9, "전체 최대 q 가 ^SX7E 가 아니게 됐다"


def test_sensitivity_is_recorded_next_to_the_proxy():
    """민감도를 코드 옆에 적어 둬야 다음 사람이 모르고 쓰지 않는다."""
    import inspect

    src = inspect.getsource(DV)
    assert "76.5" in src or "민감도" in src, "프록시 옆에 민감도 기록이 없다"
