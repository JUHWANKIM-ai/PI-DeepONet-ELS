# -*- coding: utf-8 -*-
"""build_div 의 특성화 테스트 — 기념일 되풀이와 인덱싱 규약을 못박는다.

 변이 측정에서 build_div 은 **줄 커버리지 100% · 뮤턴트 킬률 40%** 였다
 (tests/test_mc_variants_price_row.py 가 끝까지 실행시키지만 거의 검증하지 않는다).
 Savoia 가 CRAP 의 약점으로 지목한 바로 그 경우다. 여기서 생존 뮤턴트를 겨냥한다.

   L59 N = int(round(ten * 365))        창 길이
   L60 divlog = np.zeros((N, 3))        모양
   L62 nev, evl = 0, []                 초기값
   L64 [] if tk in skip else ...        총수익지수 제외
   L65 qc[i] = -sum(log1p(-y))          연속 환산
   L68 while nd <= t0                   t0 **이후** 첫 기념일 (t0 당일은 밀어낸다)
   L70 while (nd - t0).days <= N        만기일 당일 포함
   L72 divlog[day - 1, i] += ...        행 인덱스와 누적
   L74 evl.append((i, tk, day, month, drop))

 div_events 는 고정 스텁으로 바꾼다 — 그쪽은 캐시 적재라 관심사가 다르고,
 스텁으로 바꾸면 모든 단정이 '정확한 값' 이 된다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import module.dividend as DV                       # noqa: E402

T0 = pd.Timestamp("2024-06-03")
TEN = 3.0
N = 1095                                            # int(round(3.0 * 365)), t0 + 1095일 = 2027-06-03
TKS = ["AAA", "BBB", "CCC"]


def stub(monkeypatch, table):
    """티커 -> [(배당락일, delta)] 로 div_events 를 고정한다. 표에 없는 티커는 호출 자체가 오류."""
    def fake(tk, t0):
        assert t0 == T0
        assert tk in table, "div_events 가 호출되지 않아야 하는 티커: {}".format(tk)
        return list(table[tk])
    monkeypatch.setattr(DV, "div_events", fake)


def anniversaries(dj, t0=T0, n=N):
    """build_div 과 **독립적으로** 기대 기념일을 계산한다 (같은 규약, 다른 코드)."""
    nd = pd.Timestamp(dj)
    while nd <= t0:
        nd = nd + pd.DateOffset(years=1)
    out = []
    while (nd - t0).days <= n:
        out.append((nd, (nd - t0).days))
        nd = nd + pd.DateOffset(years=1)
    return out


def test_window_length_and_shapes(monkeypatch):
    """N = round(ten*365) 이고 divlog 는 (N,3), qcont 는 (3,)."""
    stub(monkeypatch, {t: [] for t in TKS})
    divlog, qc, nev, evl = DV.build_div(TKS, T0, TEN)
    assert divlog.shape == (N, 3)
    assert qc.shape == (3,)
    assert divlog.dtype == np.float32 and qc.dtype == np.float32


def test_window_length_rounds_not_truncates(monkeypatch):
    """ten 이 딱 떨어지지 않으면 **반올림**이다 (버림이면 1094 가 아니라 1093 이 된다)."""
    stub(monkeypatch, {t: [] for t in TKS})
    ten = 2.9986                                    # 2.9986 * 365 = 1094.489 -> 1094
    divlog, _, _, _ = DV.build_div(TKS, T0, ten)
    assert divlog.shape[0] == 1094


def test_no_events_is_all_zero(monkeypatch):
    """배당 기록이 없으면 전부 0 이고 이벤트 수도 0, 목록도 빈다."""
    stub(monkeypatch, {t: [] for t in TKS})
    divlog, qc, nev, evl = DV.build_div(TKS, T0, TEN)
    assert nev == 0 and evl == []
    assert not divlog.any() and not qc.any()


def test_skip_set_bypasses_div_events(monkeypatch):
    """skip 티커는 div_events 를 **부르지 않고** 배당 0 으로 둔다 (총수익지수에 이력 파일이 없다)."""
    stub(monkeypatch, {"AAA": [(pd.Timestamp("2024-03-15"), 0.01)], "CCC": []})
    tks = ["AAA", "BBB", "CCC"]                     # BBB 는 표에 없으므로 호출되면 스텁이 터진다
    divlog, qc, nev, evl = DV.build_div(tks, T0, TEN, skip={"BBB"})
    assert qc[1] == 0.0
    assert not divlog[:, 1].any()
    assert all(e[0] != 1 for e in evl)


def test_zero_div_is_the_default_skip(monkeypatch):
    """기본 skip 은 ZERO_DIV — ^GDAXI 가 들어와도 FileNotFoundError 가 나지 않는다."""
    assert "^GDAXI" in DV.ZERO_DIV
    stub(monkeypatch, {"AAA": [], "CCC": []})
    divlog, qc, nev, evl = DV.build_div(["AAA", "^GDAXI", "CCC"], T0, TEN)
    assert qc[1] == 0.0 and nev == 0


def test_qcont_is_negative_sum_of_log_drops(monkeypatch):
    """qcont[i] = -sum log(1-delta) — 연 배당을 연속 수익률로 환산한 값."""
    ys = [0.01, 0.02, 0.005]
    evs = [(pd.Timestamp("2024-03-15"), y) for y in ys]
    stub(monkeypatch, {"AAA": evs, "BBB": [], "CCC": []})
    _, qc, _, _ = DV.build_div(TKS, T0, TEN)
    assert qc[0] == pytest.approx(-sum(np.log1p(-y) for y in ys), rel=1e-6)
    assert qc[1] == 0.0 and qc[2] == 0.0


def test_event_on_issue_date_rolls_to_next_year(monkeypatch):
    """배당락일이 발행일 당일이면 **다음 해**부터 센다 (t0 당일에 하락을 넣지 않는다).

     t0 + 1095일 = 2027-06-03 이므로 기념일은 365 · 730 · 1095 세 번이고 마지막이 경계에 딱 걸린다."""
    stub(monkeypatch, {"AAA": [(T0, 0.01)], "BBB": [], "CCC": []})
    divlog, _, nev, evl = DV.build_div(TKS, T0, TEN)
    days = [e[2] for e in evl]
    assert days == [365, 730, 1095]
    assert nev == 3
    assert divlog[1094, 0] == pytest.approx(np.log1p(-0.01), rel=1e-6)
    assert divlog[-1, 0] != 0.0                      # 만기일 당일은 포함


def test_chain_stops_before_passing_maturity(monkeypatch):
    """기념일 사슬은 만기를 넘기는 순간 멈춘다.

     배당락일이 발행일보다 하루 늦은 월·일이면 (2023-06-04 vs t0 2024-06-03) 첫 기념일이
     **1년차 안**(경과 1일)에 들어오고, 사슬은 1 · 366 · 731 로 끝난다 — 1096 은 N=1095 를 넘어 빠진다."""
    dj = pd.Timestamp("2023-06-04")
    stub(monkeypatch, {"AAA": [(dj, 0.01)], "BBB": [], "CCC": []})
    divlog, _, nev, evl = DV.build_div(TKS, T0, TEN)
    assert [e[2] for e in evl] == [1, 366, 731]
    assert nev == 3
    assert divlog[0, 0] != 0.0                       # 경과 1일 -> 첫 행
    assert not divlog[731:, 0].any()                 # 731 일 이후로는 없다


def test_drop_lands_on_day_minus_one_row(monkeypatch):
    """divlog 행 인덱스는 day-1 이다 (day 는 t0 로부터의 경과일수, 1부터)."""
    dj = pd.Timestamp("2024-03-15")
    y = 0.013
    stub(monkeypatch, {"AAA": [(dj, y)], "BBB": [], "CCC": []})
    divlog, _, _, _ = DV.build_div(TKS, T0, TEN)
    exp = anniversaries(dj)
    assert len(exp) >= 2
    nz = np.nonzero(divlog[:, 0])[0]
    assert list(nz) == [d - 1 for _, d in exp]
    for _, d in exp:
        assert divlog[d - 1, 0] == pytest.approx(np.log1p(-y), rel=1e-6)


def test_same_day_events_accumulate(monkeypatch):
    """같은 날로 모이는 두 이벤트는 **더해진다** (덮어쓰지 않는다)."""
    y1, y2 = 0.01, 0.02
    evs = [(pd.Timestamp("2024-03-15"), y1), (pd.Timestamp("2023-03-15"), y2)]
    stub(monkeypatch, {"AAA": evs, "BBB": [], "CCC": []})
    divlog, _, nev, evl = DV.build_div(TKS, T0, TEN)
    day = anniversaries(evs[0][0])[0][1]
    assert anniversaries(evs[1][0])[0][1] == day, "두 이벤트가 같은 날에 모여야 하는 설정"
    assert divlog[day - 1, 0] == pytest.approx(np.log1p(-y1) + np.log1p(-y2), rel=1e-6)
    assert nev == 2 * len(anniversaries(evs[0][0]))


def test_event_list_records_slot_ticker_day_month_drop(monkeypatch):
    """evl 항목은 (슬롯, 티커, 경과일수, 월, 로그하락) 이다."""
    dj = pd.Timestamp("2024-03-15")
    y = 0.02
    stub(monkeypatch, {"AAA": [], "BBB": [(dj, y)], "CCC": []})
    _, _, _, evl = DV.build_div(TKS, T0, TEN)
    exp = anniversaries(dj)
    assert len(evl) == len(exp)
    for (slot, tk, day, month, drop), (nd, d) in zip(evl, exp):
        assert slot == 1 and tk == "BBB"
        assert day == d and month == nd.month
        assert drop == pytest.approx(float(np.log1p(-y)), rel=1e-6)


def test_slots_are_independent(monkeypatch):
    """세 슬롯은 서로 섞이지 않는다 — 다른 자산의 배당이 다른 열에 들어간다."""
    stub(monkeypatch, {"AAA": [(pd.Timestamp("2024-03-15"), 0.01)],
                       "BBB": [],
                       "CCC": [(pd.Timestamp("2024-01-10"), 0.02)]})
    divlog, qc, _, _ = DV.build_div(TKS, T0, TEN)
    assert divlog[:, 0].any() and divlog[:, 2].any()
    assert not divlog[:, 1].any()
    assert qc[0] > 0 and qc[2] > 0 and qc[1] == 0.0
    assert set(np.nonzero(divlog[:, 0])[0]).isdisjoint(np.nonzero(divlog[:, 2])[0])
