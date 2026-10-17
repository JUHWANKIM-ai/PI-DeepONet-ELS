# -*- coding: utf-8 -*-
"""프라이서 속도·일치도 표의 기준은 정본 경로 수여야 한다.

 모델은 100,000 경로 라벨(`mc_B`)로 학습했다. 그런데 노트북 3 의 속도 절이 비교·일치도 기준을
 40,000 경로 가격으로 잡고 있었다. 그러면 (1) 속도향상이 2.4배 과소평가되고,
 (2) 대리모델을 **자기가 배운 적 없는 더 시끄러운 가격**과 비교해 불일치가 부풀려진다.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm      # noqa: E402

REF = "MC 100,000 paths"
DON_REQ = "DeepONet (200-product request)"
DON_BOOK = "DeepONet (full book,"          # 건수는 OOS 크기에서 온다 — 하드코딩하면 낡는다

pytestmark = pytest.mark.skipif(not fm.stat("cat3_speed").exists(),
                                reason="cat3_speed.csv 가 없다 (노트북 3 미실행)")


def table():
    return pd.read_csv(fm.stat("cat3_speed")).set_index("engine")


def test_speedup_is_relative_to_the_canonical_path_count():
    S = table()
    ref = S.at[REF, "ms_per_product"]
    dons = [e for e in S.index if e == DON_REQ or e.startswith(DON_BOOK)]
    assert len(dons) == 2, dons
    for e in dons:
        assert S.at[e, "speedup"] == pytest.approx(ref / S.at[e, "ms_per_product"], rel=1e-6)


def test_both_path_counts_are_still_reported():
    """40,000 경로도 표에 남긴다 — 경로 수가 지연에 선형이라는 증거다."""
    S = table()
    assert "MC 40,000 paths" in S.index
    assert S.at[REF, "ms_per_product"] > S.at["MC 40,000 paths", "ms_per_product"]


def test_agreement_records_which_prices_it_was_measured_against():
    """일치도 기준을 표에 적어 둬야 한다 — 숫자만 보면 어느 경로 수와 비교한 것인지 알 수 없다."""
    S = table()
    assert "agree_ref" in S.columns
    assert set(S.agree_ref.unique()) == {REF}


def test_full_book_label_states_the_actual_oos_size():
    """표에 적힌 건수가 실제 OOS 건수여야 한다 — 하드코딩된 29,743 이 123건 복구 후 낡았다."""
    import pandas as pd
    S = table()
    lab = [e for e in S.index if e.startswith(DON_BOOK)][0]
    n = int(lab.split(",", 1)[1].strip(" )").replace(",", ""))
    L = pd.read_csv(fm.stat("allstar_level"))
    assert n == int(L[L.group == "ALL"].n.iloc[0])
