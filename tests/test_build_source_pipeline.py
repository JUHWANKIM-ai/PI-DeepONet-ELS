# -*- coding: utf-8 -*-
"""build_source() 자체의 특성화(characterization) 테스트 — 지금 동작을 그대로 못박는다.

 기존 tests/test_build_source_structure.py 는 **순수 헬퍼만** 덮는다
 (STRUCTS / encode_barrier / structure_mask / needs_linear_pmt).
 정작 그 헬퍼들을 호출하는 49분기 오케스트레이터 build_source() 는 커버리지 1% 였다.
 여기서는 합성 raw 자료를 tmp 경로에 깔고 build_source 를 실제로 돌려
 **탈락 규칙과 파생열 규약**을 고정한다. 리팩터링의 안전망 역할이다.

 주의: save=False 를 반드시 넘긴다. 기본값 save=True 는 fm.DATA 에 els3_dataset.parquet 를
 덮어쓴다 (module/build_source.py:192-194).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm                      # noqa: E402
import module.build_source as BS                     # noqa: E402
import module.underlying as UL                       # noqa: E402

TICKERS = ["^AAA", "BBB.KS", "CCC.T"]
UDLY = ["U_A", "U_B", "U_C"]
ISU = "2024-06-03"
MAT = "2027-06-03"                                       # tenor ≈ 3.0 → TEN_LO..TEN_HI 안
NOBS = 6


def _prices(seed):
    """영업일 400개. vol180 은 60표본, corr180 도 60표본이면 충분하다."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=400)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(idx))))
    return pd.DataFrame({"close": px}, index=idx)


def _autocall_row(item, opt_type="STEP", ki=1, fair=0.95, cur="KRW"):
    return dict(ITEM_CD=item, OPT_TYPE=opt_type, KNCK_IN_YN=ki, CUR_CD=cur,
                FAIR_VALUE=fair * 10000.0, ISU_PRC_DETAIL=10000.0,
                ISU_DT=ISU, MAT_DT=MAT, SUB_START_DT="2024-05-27", SUB_END_DT="2024-05-31",
                ANL_RTRN=6.0, ISU_ORG="테스트증권", RISK_GRADE="2", PRODUCT_TYPE="ELS",
                RDMP_TYPE="상환", ACT_ISU_AMT=1.0e9, SB_RT=0.3, DV_RT=0.0,
                PRCP_GRTE_RT=0.0, KNCK_IN_GRC_PRD=0.0)


def _sched_rows(item, nobs=NOBS, barr=50.0):
    """STEP 스케줄: 행사가 0.90→0.65 하향, 지급률 선형."""
    ks = np.linspace(0.90, 0.65, nobs)
    return [dict(ITEM_CD=item, SEQ=j + 1, SCHD_TYPE=1, STRK_1=float(ks[j]),
                 PMT_1=float(0.06 * (j + 1) / 2.0), LZRD_BARR=np.nan, LZRD_PMT=np.nan,
                 LZRD_TERM=np.nan, BARR_1=barr)
            for j in range(nobs)]


@pytest.fixture
def raw_universe(tmp_path, monkeypatch):
    """4개 상품을 깐다 — 통과 1, 구조 탈락 1, 티커매핑 탈락 1, 스케줄 탈락 1."""
    cache, raw, data = tmp_path / "cache", tmp_path / "raw", tmp_path / "data"
    for d in (cache, raw, data):
        d.mkdir(parents=True)

    (raw / "udly_ticker_map.json").write_text(
        json.dumps(dict(zip(UDLY, TICKERS))), encoding="utf-8")

    # 가격은 share_underlyings_v2 에서 온다 (module.mc.returns_by_ticker -> module.underlying).
    # 종전에는 px_* 캐시를 깔았는데, 캐시는 입력이 아니라 산출물이므로 공급원이 바뀌었다.
    su = data / "share_underlyings_v2"
    su.mkdir(parents=True)
    rows = []
    for i, t in enumerate(TICKERS):
        px = _prices(i)
        fn = "u{}.csv".format(i)
        px.rename(columns={"close": "Close_Adj"}).rename_axis("Date").to_csv(su / fn)
        rows.append({"가격티커(yfinance)": t, "파일": fn,
                     "ETF티커(yfinance)": "", "ETF_파일": ""})
    pd.DataFrame(rows).to_csv(su / "master.csv", index=False, encoding="utf-8-sig")
    # 모듈 캐시를 비워야 실제 master 가 새지 않는다. **monkeypatch 로** 비워야 테스트가 끝날 때
    # 원래 값이 돌아온다 — 직접 대입하면 합성 마스터(티커 3개)가 다음 테스트로 새어 나가
    # `KeyError: '^KS200'` 을 일으킨다.
    monkeypatch.setattr(UL, "_MASTER", None)
    monkeypatch.setattr(UL, "_PX", {})

    kidx = pd.bdate_range("2023-01-02", periods=400)
    (raw / "rates").mkdir()
    pd.DataFrame({"call": 3.0, "m3": 3.3, "y10": 3.6}, index=kidx).to_parquet(
        raw / "rates" / "krw_curve.parquet")      # 금리는 원천 (fm.rate)

    ac = [_autocall_row("P_OK"),
          _autocall_row("P_BADSTRUCT", opt_type="OTHER"),     # structure_mask 에서 제외
          _autocall_row("P_NOMAP"),                           # UDLY_ID 가 매핑에 없음
          _autocall_row("P_SHORTSCHED")]                      # 스케줄 1행 → nobs<2
    pd.DataFrame(ac).to_csv(raw / "LAKE_V2_DART_AUTO_CALL.csv", index=False)

    sc = (_sched_rows("P_OK") + _sched_rows("P_BADSTRUCT") + _sched_rows("P_NOMAP")
          + _sched_rows("P_SHORTSCHED", nobs=1))
    pd.DataFrame(sc).to_csv(raw / "LAKE_V2_DART_SCHD_INFO.csv", index=False)

    ud = []
    for item, uids in [("P_OK", UDLY), ("P_BADSTRUCT", UDLY), ("P_SHORTSCHED", UDLY),
                       ("P_NOMAP", ["U_A", "U_B", "U_MISSING"])]:
        ud += [dict(ITEM_CD=item, UDLY_ID=u) for u in uids]
    pd.DataFrame(ud).to_csv(raw / "LAKE_V2_DART_UDLY_INFO.csv", index=False)

    monkeypatch.setattr(fm, "CACHE", cache)
    monkeypatch.setattr(fm, "RAW", raw)
    monkeypatch.setattr(fm, "DATA", data)
    return tmp_path


def test_only_eligible_products_survive(raw_universe):
    df = BS.build_source(save=False, verbose=False)
    assert list(df["item"]) == ["P_OK"], (
        "통과해야 할 상품은 P_OK 하나다. 실제: {}".format(list(df["item"])))


def test_build_source_assigns_mc_seed(raw_universe):
    """mc_seed 는 발행일 정렬 후 행 인덱스 — MC 를 돌리지 않아도 정해지는 결정적 값이다.

     종전에는 module.mc_shards.combine() 이 붙였다. 그 샤드 캐시가 없는 원천 전용 재현에서는
     여기서 나와야 한다."""
    df = BS.build_source(save=False, verbose=False)
    assert "mc_seed" in df.columns
    assert (df.mc_seed.values == np.arange(len(df))).all()
    assert df.mc_seed.dtype == np.int32


def test_save_false_writes_nothing(raw_universe):
    BS.build_source(save=False, verbose=False)
    assert not (fm.DATA / "els3_dataset.parquet").exists(), \
        "save=False 인데 데이터셋을 썼다 — 정본 parquet 를 덮어쓸 위험"


def test_derived_columns_are_internally_consistent(raw_universe):
    r = BS.build_source(save=False, verbose=False).iloc[0]
    assert r.sig1 <= r.sig2 <= r.sig3, "sig1..3 은 오름차순 정렬 규약"
    assert r.rho12 <= r.rho13 <= r.rho23, "rho 3개도 오름차순 정렬 규약"
    assert r.sig_min == pytest.approx(r.sig1) and r.sig_max == pytest.approx(r.sig3)
    assert r.stepdown == pytest.approx(r.Kfirst - r.K), "stepdown = 첫 행사가 − 마지막 행사가"
    assert r.b_over_k == pytest.approx(r.B / r.K), "b_over_k = 배리어 / 마지막 행사가"
    assert r.cpn_spread == pytest.approx(r.coupon - r.r), "cpn_spread = 쿠폰 − 만기일치 무위험금리"
    assert r.nobs == NOBS and r.ki_yn == 1 and r.opt_type == "STEP"
    assert r.B == pytest.approx(0.50), "KNCK_IN_YN=1 이면 BARR_1/100"
    assert list(sorted([r.udl1, r.udl2, r.udl3])) == sorted(TICKERS), "udl 은 정렬 저장"


def test_curve_node_summaries_match_the_ten_nodes(raw_universe):
    r = BS.build_source(save=False, verbose=False).iloc[0]
    u = np.array([r[f"u{j}"] for j in range(10)])
    assert r.curve_level == pytest.approx(u.mean())
    assert r.curve_slope == pytest.approx(u[9] - u[0])
    assert r.curve_curv == pytest.approx(2 * u[4] - u[0] - u[9])


def test_drop_reasons_are_only_observable_on_stdout(raw_universe, capsys):
    """탈락 사유 분류는 반환값에 없고 verbose 출력에만 있다 — 현재 인터페이스의 한계를 고정."""
    BS.build_source(save=False, verbose=True)
    out = capsys.readouterr().out
    assert "티커 매핑 실패" in out, "P_NOMAP 은 티커 매핑 실패로 떨어져야 한다"
    assert "스케줄" in out, "P_SHORTSCHED 는 스케줄 사유로 떨어져야 한다"
    assert "예외:" not in out, "예외로 삼켜진 행이 있다 — bare except 가 버그를 감추고 있다"
