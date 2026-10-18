# -*- coding: utf-8 -*-
"""원천 전용 재현의 판정 — 지수전용 부분집합 대조.

 **왜 부분집합인가.** 가격 계열이 `px_*` 캐시(배당 역조정)에서 `Close_Adj`(액면분할만)로
 바뀌었다. 개별주식은 두 계열이 실제로 다르므로(삼성전자 2015-01-02: 20,419 vs 26,600)
 라벨이 달라지는 것이 **정상**이다. 반면 지수는 두 계열이 거의 같다(^KS200 수익률 상관 0.99994).
 그래서 지수만으로 구성된 상품에서는 라벨이 재현돼야 하고, 그 집합이 판정 대상이다.

 **가중치는 비트 단위로 대조하지 않는다.** 시드가 0~4 로 같아도 라벨이 바뀌었으므로 텐서는
 반드시 다르다. 파일 개수·이름 규약의 일치와 성능 재현으로 본다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                              # noqa: E402
from generate.verify_raw_run import (TIGHT, IDX, BASE, EXP_MAD, GATE,   # noqa: E402
                                  METRICS, compare, model_verdict, verdict)

needs_baseline = pytest.mark.skipif(
    not (BASE / "cache" / "mcvar" / "mc_variants_lseg.parquet").exists(),
    reason="구본 스냅샷이 없다 (Task 8 승인 후 삭제됨)")
# model_verdict 는 기준선이 삭제된 뒤에도 고정 픽스처로 동작한다 — 그 경로를 재는 테스트는
# 기준선이 아니라 픽스처 유무로 건너뛰어야 한다. 아니면 판정 테스트가 통째로 휴면한다.
needs_fixture = pytest.mark.skipif(
    not (Path(__file__).resolve().parent / "fixtures" / "raw_run_verdict.json").exists(),
    reason="판정 근거 픽스처가 없다")
needs_models = pytest.mark.skipif(
    not any((fm.RESULT / "models").glob("*.pt")),
    reason="신본 가중치가 아직 없다 (노트북 2 미실행)")


def test_tight_is_a_subset_of_idx():
    assert TIGHT < IDX
    assert IDX - TIGHT == {"^HSI", "^STOXX50E"}


def test_subsets_are_disjoint_and_sized():
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet")
    m = d.udl1.isin(IDX) & d.udl2.isin(IDX) & d.udl3.isin(IDX)
    assert 50_000 < int(m.sum()) < 58_790          # 지수전용 56,162건 전후


@needs_baseline
def test_compare_reports_three_subsets():
    T = compare()
    assert list(T.subset) == ["TIGHT", "IDX", "OTHER"]
    assert T.n.sum() == len(pd.read_parquet(fm.DATA / "els3_dataset.parquet"))


@needs_baseline
def test_compare_only_uses_items_present_in_both():
    """신본 전용 123건을 섞으면 상관이 의미를 잃는다 — 공통 항목만 쓰고 개수를 밝힌다."""
    T = compare()
    assert (T.n_cmp <= T.n).all()
    assert T.n_cmp.sum() < T.n.sum()
    assert T.n_only_new.sum() > 0                  # 새로 살아난 상품을 따로 보고한다


@needs_baseline
def test_compare_counts_products_lost_as_well_as_gained():
    """**구본에만 있던 상품**도 세야 한다 — 이 브랜치가 막으려는 위험이 바로 조용한 상품 손실이다.

     종전 compare() 는 신본 데이터셋을 기준으로 구본을 조인해서, 신본에서 사라진 상품을
     구조적으로 볼 수 없었다. 실측 0 이지만 0 임을 **보여야** 의미가 있다."""
    T = compare()
    assert "n_only_old" in T.columns
    assert T.n_only_old.sum() == 0, "구본에만 있는 상품이 생겼다"


@needs_baseline
def test_only_new_counts_dataset_membership_not_label_presence():
    """n_only_new 는 '새로 들어온 상품' 이어야 한다 — 라벨 결측과 섞으면 수가 맞지 않는다.

     종전에는 구본 라벨이 NaN 이던 18행까지 더해 141 로 나왔는데, 실제 신규 유입은 123 이다."""
    T = compare()
    o = pd.read_parquet(BASE / "datasets" / "els3_dataset.parquet", columns=["item"])
    n = pd.read_parquet(fm.DATA / "els3_dataset.parquet", columns=["item"])
    assert T.n_only_new.sum() == len(set(n.item) - set(o.item)) == 123


@needs_baseline
def test_compare_columns_are_pinned():
    T = compare()
    assert list(T.columns) == ["subset", "n", "n_cmp", "n_only_new", "n_only_old", "mc_corr",
                               "mc_mad", "mc_mad_se", "mc_mad_exp", "mc_bias",
                               "px_n", "px_mad", "px_se", "px_hi", "px_bias",
                               "r2_old", "r2_new", "mape_old", "mape_new", "frozen"]


@needs_baseline
def test_label_diff_cannot_test_reproduction():
    """구본과 신본은 공통난수가 아니다 — mc_seed 가 행 번호라 123건 유입으로 98% 가 바뀌었다.

     그래서 mc_mad 는 '재현 실패'가 아니라 독립 MC 잡음 두 개의 합이고, 기대값이 EXP_MAD 다.
     판정은 공통난수로 잡음이 상쇄된 px_mad 로 해야 한다."""
    o = pd.read_parquet(BASE / "datasets" / "els3_dataset.parquet",
                        columns=["item", "mc_seed"]).set_index("item")
    n = pd.read_parquet(fm.DATA / "els3_dataset.parquet",
                        columns=["item", "mc_seed"]).set_index("item")
    J = o.join(n, how="inner", lsuffix="_o", rsuffix="_n")
    assert (J.mc_seed_o == J.mc_seed_n).mean() < 0.05          # 실측 1.8%
    assert 6.0 < EXP_MAD < 8.5                                  # 40k + 100k 잡음 합 7.18원


@needs_baseline
def test_px_effect_is_noise_cancelled():
    """공통난수 측정에서 TIGHT 지수의 가격계열 효과는 사실상 0 이어야 한다."""
    T = compare().set_index("subset")
    assert T.at["TIGHT", "px_mad"] < 1.0
    assert T.at["OTHER", "px_mad"] > T.at["IDX", "px_mad"] > T.at["TIGHT", "px_mad"]


@needs_baseline
def test_verdict_judges_on_px_mad_not_label_diff():
    T = compare()
    V = verdict(T)
    assert set(V) == {"TIGHT", "IDX", "OTHER"}
    assert V["TIGHT"][0] is True                                # 0.10 +- 0.06, 상한도 5원 밑
    assert V["OTHER"][0] is None                                # 기준 없음
    assert set(GATE) == {"TIGHT", "IDX"}


@needs_baseline
def test_verdict_uses_the_interval_not_the_point_estimate():
    """점추정이 기준 밑이어도 구간 상한이 넘으면 '통과' 가 아니다.

     IDX 는 px_mad 8.15 +- 1.67 (n=89) 로 점추정은 10원 밑이지만 95% 단측 상한이 10원을
     넘는다. 분포가 우측으로 길어(최대 135원) 정규 SE 로도 과소추정이다 — 동전던지기를
     '통과' 로 적으면 안 된다."""
    T = compare().set_index("subset")
    V = verdict(compare())
    assert T.at["IDX", "px_mad"] < GATE["IDX"] < T.at["IDX", "px_hi"]
    assert V["IDX"][0] is None, "구간 상한이 기준을 넘는데 통과로 판정했다"


@needs_baseline
def test_label_gap_residual_is_reported_not_explained_away():
    """mc_mad 와 잡음기대의 차이는 SE 에 비해 크다 — '대략 일치' 로 뭉개면 안 된다.

     TIGHT: mc_mad 8.33 (n_cmp 6,659, SE ~0.1) vs EXP_MAD 7.18 -> 1.15원이 설명되지 않는다.
     게다가 EXP_MAD 의 두 입력은 40만 경로 준참값 대비 측정치라 기준 자체의 분산을 이미
     품고 있어 기대값을 **과대**추정한다. 즉 미설명분은 하한이다."""
    T = compare().set_index("subset")
    r = T.loc["TIGHT"]
    resid = r.mc_mad - r.mc_mad_exp
    assert resid > 3 * r.mc_mad_se                      # 잡음으로 설명되지 않는다
    assert "미설명" in verdict(compare())["TIGHT"][1]


@needs_baseline
@needs_models
def test_weight_inventory_matches_baseline():
    """개수와 이름 규약은 일치해야 한다 (텐서 값은 라벨이 달라 다를 수밖에 없다).

     **기준선 스냅샷이 삭제되면 건너뛴다.** 이 요구사항 자체는 사라지지 않는다 —
     tests/test_notebook_model_inventory.py::test_weight_count_matches_the_contract 가
     기대 집합(7변형 x 5시드 x 4폴드 = 140)을 **구성해서** 대조하므로 스냅샷보다 강하다.
     이 테스트는 스냅샷이 있던 동안 그 구성이 실제 구본과 같았음을 확인한 역할이다."""
    # **가중치만** 센다. result/models/ 에는 `.trained_label_stamp.json`(어떤 라벨로 학습했는지)
    # 같은 메타데이터도 함께 있고, 그건 가중치 목록의 일부가 아니다.
    def names(d):
        return {p.name for p in list(Path(d).glob("*.pt")) + list(Path(d).glob("*.pkl"))}
    assert names(fm.RESULT / "models") == names(BASE / "models")


@needs_baseline
def test_model_verdict_reports_its_own_power():
    """|z| <= 2 는 '구별되지 않는다' 이지 '같다' 가 아니다 — 검출 가능한 최소 차이를 밝혀야 한다.

     시드 5개 x 2 로 결합 SE 가 R2 0.0072 이므로 이 검정은 0.014 보다 작은 하락을 잡지 못한다.
     실측 하락이 0.0064 이므로 **관측된 하락의 두 배**가 검출 한계다. 게다가 시드 SE 는 학습
     난수만 재고 라벨 교체는 다섯 시드에 공통인 고정효과라, 검정 대상에 구조적으로 눈을 감는다."""
    V = model_verdict()
    for m, (ok, z, why) in V.items():
        assert "검출한계" in why, "검정력을 밝히지 않는다: {}".format(why)
    assert all(v[0] for v in V.values())


@needs_baseline
def test_model_verdict_flags_the_direction_anomaly():
    """R2 가 **떨어진** 것은 예측 방향과 반대다 — 10만 경로 라벨은 4만보다 덜 시끄러워
     달성 가능한 R2 가 올라야 한다. 구별 불가라도 열린 항목으로 남겨야 한다."""
    _, z, why = model_verdict()["R2"]
    assert z < 0                                    # 신본이 낮다
    assert "예상과 반대" in why


@needs_baseline
def test_model_verdict_uses_a_two_sample_test():
    """구본 성능도 시드 5개의 **추정값**이다 — 참값처럼 쓰면 판정이 한쪽으로 기운다.

     계획서는 구본 SE 만으로 +-2SE 를 재라고 했다. 그 기준으로는 MAPE 차이 +0.034 가
     구간 밖(구본 SE 0.0155)이지만, 신본의 SE 가 0.0378 이므로 결합 SE 0.041 로는 0.84 SE 다.
     두 표본 모두의 분산을 쓰는 것이 옳다."""
    V = model_verdict()
    assert set(V) == set(METRICS)
    for m, (ok, z, _) in V.items():
        assert np.isfinite(z)
        assert ok == (abs(z) <= 2.0)
    # 결합 SE 는 구본 SE 보다 반드시 크거나 같다 -> 판정이 더 관대해진다
    assert all(ok for ok, _, _ in V.values()), "모델 성능이 재현되지 않았다"




@needs_fixture
def test_model_verdict_states_the_population_mismatch():
    """두 팔이 **다른 상품 집합**에서 측정됐다 — 완벽한 재현이라도 0 이 아니다.

     구본 n=29,743, 신본 n=29,916 으로 173건이 다르고(^SX7E 신규 포함), 시드 SE 는 그 차이를
     전혀 설명하지 못한다. 그러므로 |z| <= 2 는 재현 검정이 아니라 '서로 다른 두 양이
     학습 잡음보다 덜 다르다' 는 진술이다. 그 사실이 출력에 있어야 한다."""
    V = model_verdict()
    for m, (ok, z, why) in V.items():
        assert "n " in why or "모집단" in why, "모집단 불일치를 밝히지 않는다: {}".format(why)


@needs_fixture
def test_model_verdict_reports_an_80_percent_power_floor():
    """2·SE 는 검정력 50% 짜리 바닥이다 — 낙관적인 수치를 적으면 '구별 불가' 를 과독하게 만든다."""
    V = model_verdict()
    _, _, why = V["R2"]
    assert "80%" in why or "2.8" in why, "검정력 기준을 밝히지 않는다: {}".format(why)


@needs_fixture
def test_model_verdict_marks_a_frozen_arm():
    """기준선이 사라지면 왼쪽 팔은 **측정이 아니라 기록**이다 — compare() 는 frozen 을 알리는데
     model_verdict 는 조용히 픽스처로 바꿔치기하고 있었다."""
    import generate.verify_raw_run as V
    import inspect
    assert "frozen" in inspect.getsource(V.model_verdict)
