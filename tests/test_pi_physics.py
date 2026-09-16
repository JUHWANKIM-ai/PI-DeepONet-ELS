# -*- coding: utf-8 -*-
"""PI-DeepONet 물리항 검증.

 핵심은 마지막 두 테스트다: PI 의 경계조건이 **실측 MC 페이오프 블록**
 (module.mc_engine.payoff_from_path_summary_t, 이론가를 실제로 만든 코드) 과
 4구조(STEP/LIZARD × 낙인/노낙인) 전부에서 수치적으로 일치하는지 직접 대조한다.
 물리항이 MC 규약과 어긋나면 앵커는 '틀린 이론가'로 수렴하므로 이게 가장 중요한 가드다."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from module.mc_engine import payoff_from_path_summary_t
from module.networks import PIStateDeepONet
from module.pi_collocation import PICollocationSampler
from module.pi_physics import (continuation_relation, event_payoff, event_tau,
                               feasible_ki_states, ki_transition_mask, pde_residual,
                               terminal_payoff)
from module.pi_train import COMPONENTS, PREDECESSOR, issuance_state, state_norm


# ===== 경계조건 =====
def test_terminal_payoff_covers_hit_and_ki_branches():
    spot = torch.tensor([[.95, .90, .85], [.70, .80, .90]])
    strike = torch.tensor([.80, .80])
    pmt = torch.tensor([.12, .12])
    # 만기 행사 성공 -> 1+pmt / 실패 -> KI 면 worst, 아니면 원금 보전
    assert torch.allclose(terminal_payoff(spot, strike, pmt, torch.tensor([0., 0.])),
                          torch.tensor([1.12, 1.00]))
    assert torch.allclose(terminal_payoff(spot, strike, pmt, torch.tensor([1., 1.])),
                          torch.tensor([1.12, .70]))


def test_event_priority_is_regular_then_lizard():
    spot = torch.tensor([[.95, .90, .85], [.85, .84, .83], [.75, .80, .85]])
    running_min = torch.tensor([[.80, .82, .81], [.82, .81, .80], [.75, .78, .80]])
    value, active, regular, lizard = event_payoff(
        spot, running_min, strike=torch.tensor([.80, .90, .90]), pmt=torch.tensor([.10, .10, .10]),
        lz_barrier=torch.tensor([.79, .79, .79]), lz_pmt=torch.tensor([.05, .05, .05]))
    assert regular.tolist() == [True, False, False]
    assert lizard.tolist() == [False, True, False]
    assert active.tolist() == [True, True, False]
    assert torch.allclose(value[:2], torch.tensor([1.10, 1.05]))


def test_ki_transition_uses_running_minimum_and_contract_barrier():
    running_min = torch.tensor([[.70, .80, .90], [.90, .90, .90]])
    assert ki_transition_mask(running_min, torch.tensor([.75, .75])).tolist() == [True, False]


def test_event_tau_uses_exact_observation_number():
    tenor = torch.tensor([3.0, 2.0]); nobs = torch.tensor([6, 4]); j = torch.tensor([0, 2])
    # 관측 j+1 회차 직후의 잔존만기
    assert torch.allclose(event_tau(tenor, nobs, j), torch.tensor([2.5, .5]))


def test_feasible_ki_never_allows_flag_zero_below_barrier():
    running = torch.tensor([[.60, .80, .90], [.80, .85, .90]])
    barrier = torch.tensor([.70, .70])
    assert feasible_ki_states(running, barrier, torch.tensor([0., 0.])).tolist() == [1., 0.]


def test_non_redemption_continuation_is_value_continuous():
    before = torch.tensor([.9, 1.1]); after = torch.tensor([.9, 1.0])
    loss = continuation_relation(before, after, torch.tensor([True, False]))
    assert torch.allclose(loss, torch.tensor(0.0))


# ===== PDE 잔차 =====
def test_pde_residual_is_finite_and_backpropagates():
    n = 4
    s = torch.full((n, 3), .9); tau = torch.full((n, 1), 1.0)
    sig = torch.full((n, 3), .2); corr = torch.eye(3).repeat(n, 1, 1); rate = torch.full((n,), .03)
    residual = pde_residual(lambda x, t: (x.square().sum(1) + t[:, 0].square()), s, tau, sig, corr, rate)
    assert residual.shape == (n,) and torch.isfinite(residual).all()
    residual.square().mean().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(tau.grad).all()


def test_pde_residual_matches_closed_form_on_quadratic_value():
    """V = sum(Si^2) + tau^2 에 대해 잔차를 해석적으로 계산해 대조 (2차 도함수 경로 검증).
     V_tau=2tau, V_Si=2Si, V_SiSj=2*delta_ij →
       resid = -2tau + 2*sum(r*Si^2) + sum_i rho_ii*sig_i^2*Si^2 - r*V   (교차항은 gij=0)"""
    n = 3
    s = torch.tensor([[.8, .9, 1.0], [1.1, .7, .95], [.6, 1.2, .85]])
    tau = torch.tensor([[1.0], [.5], [2.0]])
    sig = torch.tensor([[.2, .25, .3], [.15, .2, .35], [.3, .1, .2]])
    corr = torch.eye(3).repeat(n, 1, 1)
    rate = torch.tensor([.03, .04, .02])
    got = pde_residual(lambda x, t: (x.square().sum(1) + t[:, 0].square()), s, tau, sig, corr, rate)
    s0 = s.detach(); t0 = tau.detach()[:, 0]
    V = s0.square().sum(1) + t0.square()
    want = (-2 * t0
            + 2 * (rate[:, None] * s0.square()).sum(1)
            + (sig.square() * s0.square()).sum(1)
            - rate * V)
    assert torch.allclose(got, want, atol=1e-5), f"{got} != {want}"


# ===== 망 · collocation · 학습 규약 =====
def test_pi_network_output_shape_and_finite_gradients():
    net = PIStateDeepONet(7, 51, 16)
    out = net.V(torch.randn(5, 10), torch.randn(5, 7), torch.randn(5, 51),
                torch.randn(5, PIStateDeepONet.state_dim))
    assert out.shape == (5,)
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def _fake_D(n=8):
    """CON 레이아웃 (n,51): [0:12]strk [12:24]pmt [24:36]lz_barr [36:48]lz_pmt 48=B 49=coupon 50=tenor."""
    con = np.zeros((n, 51), dtype="float32")
    con[:, :12] = np.linspace(.95, .75, 12)
    con[:, 12:24] = np.linspace(.02, .12, 12)
    con[:, 24:36] = 1.2                               # 리자드 중립(발동 불가)
    con[:4, 24] = .70; con[:4, 36] = .03              # 앞 4개만 1회차 리자드 유효
    con[:, 48] = np.r_[np.full(4, .65), np.full(4, 1.2)]   # 낙인 4 + 노낙인 4
    con[:, 50] = 3.
    return SimpleNamespace(
        CON=con, iBARR=48, iKlast=11, TEN=np.full(n, 3., dtype="float32"),
        ml=pd.DataFrame({"nobs": np.full(n, 6)}),
        ITEM=np.array([f"i{x}" for x in range(n)]),
        VC=np.tile(np.array([.2, .25, .3, .5, .4, .45, .25], "float32"), (n, 1)),
        R=np.full(n, .03, dtype="float32"), n=n)


def test_collocation_never_uses_validation_or_oos_contracts():
    D = _fake_D()
    s = PICollocationSampler(D, np.arange(3), seed=0)
    for sample in (s.sample_interior, s.sample_terminal):
        idx, state = sample(50, "cpu")
        assert set(idx).issubset({0, 1, 2})
        assert state.shape == (50, PIStateDeepONet.state_dim)
        s.assert_train_only(idx)
    with pytest.raises(AssertionError):
        s.assert_train_only(np.array([7]))          # 학습 밖 계약이면 실패해야 한다


def test_event_sampler_uses_exact_event_time_and_feasible_ki_state():
    D = _fake_D()
    s = PICollocationSampler(D, np.arange(6), seed=0)
    idx, j, pre, post = s.sample_event(100, "cpu")
    want = event_tau(torch.tensor(D.TEN[idx]), torch.tensor(D.ml.iloc[idx].nobs.to_numpy()),
                     torch.tensor(j))
    assert torch.allclose(pre[:, 6], want) and torch.allclose(post[:, 6], want)
    assert torch.all(pre[:, 8].eq(1)) and torch.all(post[:, 8].eq(0))    # pre/post 플래그
    barr = torch.tensor(D.CON[idx, D.iBARR])
    impossible = (pre[:, 3:6].amin(1) < barr) & pre[:, 7].eq(0)
    assert not impossible.any(), "배리어 아래인데 KI=0 인 불가능 상태가 생성됨"


def test_event_sampler_handles_infeasible_lizard_region():
    D = _fake_D()
    D.CON[0, :12] = 1.25; D.CON[0, 0] = .65; D.CON[0, 24] = .70   # 리자드 발동 불가 배치
    s = PICollocationSampler(D, np.array([0]), seed=3)
    for _ in range(20):
        _, _, pre, post = s.sample_event(100, "cpu")
        assert torch.isfinite(pre).all() and torch.isfinite(post).all()


def test_ki_boundary_sampler_keeps_only_knock_in_contracts():
    D = _fake_D()
    s = PICollocationSampler(D, np.arange(8), seed=1)
    idx, s0, s1 = s.sample_ki_boundary(200, "cpu")
    assert len(idx) and (D.CON[idx, D.iBARR] < 1.).all()   # 노낙인(B>=1)은 제외
    assert torch.all(s0[:, 7].eq(0)) and torch.all(s1[:, 7].eq(1))


def test_issuance_state_marks_no_ki_contracts_as_ki():
    D = _fake_D()
    st = issuance_state(D, np.arange(8))
    assert st.shape == (8, PIStateDeepONet.state_dim)
    assert np.allclose(st[:, :6], 1.)                  # 발행시점 S=M=1
    assert st[:4, 7].tolist() == [0, 0, 0, 0]          # 낙인(B<1) -> KI 미발생
    assert st[4:, 7].tolist() == [1, 1, 1, 1]          # 노낙인(B>=1) -> MC 규약상 항상 worst


def test_state_norm_is_invertible_on_price_block():
    x = torch.tensor([[.8, .9, 1.0, .7, .8, .9, 3.0, 1., 0., .5]])
    y = state_norm(x, 3.0, .15)
    assert torch.allclose(y[:, :6] * .25 + 1., x[:, :6])
    assert torch.allclose(y[:, 6], torch.zeros(1))     # tau == 평균 -> z=0
    assert torch.allclose(y[:, 7:], x[:, 7:])          # 플래그는 그대로


def test_data_only_variant_has_no_physics_and_curriculum_chain_is_consistent():
    assert COMPONENTS["data_only"] == ()
    assert "pde" in COMPONENTS["pde"]
    # 커리큘럼 사슬은 COMPONENTS 순서를 따라야 한다 (선행 항목이 부분집합)
    for variant, prev in PREDECESSOR.items():
        assert set(COMPONENTS[prev]).issubset(set(COMPONENTS[variant]))


# ===== 실측 MC 페이오프와의 직접 대조 (가장 중요) =====
def _mc_payoff(worst_obs, running_worst, final_worst, barrier, no_ki,
               lizard_barrier=float("nan"), lizard_coupon=0.0, strike=.80, pmt=.10):
    """실측 MC 페이오프 블록을 1경로·1관측으로 호출 (할인 없음: DF=1)."""
    log = lambda x: torch.log(torch.tensor(x, dtype=torch.float32))
    return payoff_from_path_summary_t(
        log([[worst_obs]]), log([[running_worst]]), log([running_worst]), log([final_worst]),
        log([strike]), torch.tensor([pmt]),
        torch.tensor([np.isfinite(lizard_barrier)]),
        log([lizard_barrier if np.isfinite(lizard_barrier) else 1.0]),
        torch.tensor([lizard_coupon]), no_ki, float(np.log(barrier)),
        torch.ones(1), torch.tensor([1]))


def test_step_structures_match_mc_terminal_payoff():
    # STEP 낙인: 만기 미상환 + 배리어 터치 -> worst-of
    pi = terminal_payoff(torch.tensor([[.65, .70, .80]]), torch.tensor([.80]),
                         torch.tensor([.10]), torch.tensor([1.]))
    assert torch.allclose(_mc_payoff(.70, .60, .65, .70, no_ki=False), pi)
    # STEP 노낙인: MC 는 B>=1 을 '만기 미상환이면 항상 worst' 로 인코딩 -> 같은 값
    assert torch.allclose(_mc_payoff(.70, .60, .65, 1.01, no_ki=True), pi)


def test_step_no_touch_keeps_principal_and_matches_mc():
    # 낙인이지만 배리어를 안 건드림 -> 원금 1.0
    pi = terminal_payoff(torch.tensor([[.75, .78, .82]]), torch.tensor([.80]),
                         torch.tensor([.10]), torch.tensor([0.]))
    assert torch.allclose(_mc_payoff(.75, .75, .75, .70, no_ki=False), pi)
    assert torch.allclose(pi, torch.tensor([1.0]))


def test_lizard_structures_match_mc_event_payoff():
    # LIZARD 낙인 / 노낙인 모두: 정규 실패 + no-touch 성립 -> 리자드 쿠폰
    for barrier, no_ki in [(.70, False), (1.01, True)]:
        mc = _mc_payoff(.75, .75, .70, barrier, no_ki, lizard_barrier=.70, lizard_coupon=.05)
        target, active, regular, lizard = event_payoff(
            torch.tensor([[.75, .80, .85]]), torch.tensor([[.75, .78, .80]]),
            torch.tensor([.80]), torch.tensor([.10]), torch.tensor([.70]), torch.tensor([.05]))
        assert active.item() and lizard.item() and not regular.item()
        assert torch.allclose(mc, target)


def test_regular_redemption_beats_lizard_in_both_implementations():
    # 정규·리자드 조건 동시 성립 -> 양쪽 모두 정규 쿠폰(0.10) 을 줘야 한다
    mc = _mc_payoff(.85, .85, .85, .70, no_ki=False, lizard_barrier=.70, lizard_coupon=.05)
    target, _, regular, lizard = event_payoff(
        torch.tensor([[.85, .88, .90]]), torch.tensor([[.85, .86, .87]]),
        torch.tensor([.80]), torch.tensor([.10]), torch.tensor([.70]), torch.tensor([.05]))
    assert regular.item() and not lizard.item()
    assert torch.allclose(mc, target) and torch.allclose(target, torch.tensor([1.10]))
