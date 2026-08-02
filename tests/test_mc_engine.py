import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from module.mc_engine import mc_daily_t, DEFAULT_DEV

# beta=[2,0,0] -> NS 제로금리 0.02 평탄, DF(t)=exp(-0.02 t)
# sigs=0 -> worst-of 경로 = exp(0.02 t) 하나로 확정 => 정답을 손으로 계산할 수 있다
BETA = np.array([2.0, 0.0, 0.0])
RATE = 0.02
SIG0 = [0.0, 0.0, 0.0]
EYE = np.eye(3)
TEN = 1.0
NP_T = 1024


def _obs_day(nobs, ten=TEN):
    N = int(round(ten * 365))
    return np.clip(np.round(np.arange(1, nobs + 1) * (ten / nobs) * 365).astype(int), 1, N)


def test_zero_vol_redeems_at_first_observation_with_pmts():
    pmts = [0.02, 0.04]
    v = mc_daily_t(SIG0, EYE, BETA, 0.5, [0.90, 0.90], TEN, n=NP_T, pmts=pmts)
    oi = _obs_day(2)[0]
    assert abs(v - (1 + pmts[0]) * np.exp(-RATE * oi / 365)) < 1e-5


def test_pmts_override_linear_coupon():
    c = 0.06
    obs_t = _obs_day(2) / 365.0
    a = mc_daily_t(SIG0, EYE, BETA, 0.5, [0.90, 0.90], TEN, n=NP_T, c=c)
    b = mc_daily_t(SIG0, EYE, BETA, 0.5, [0.90, 0.90], TEN, n=NP_T, c=0.0,
                   pmts=list(c * obs_t))
    assert abs(a - b) < 1e-6                                  # 같은 값을 명시해도 결과 동일


def test_lizard_pays_reduced_coupon_when_regular_strike_fails():
    v = mc_daily_t(SIG0, EYE, BETA, 0.5, [9.99, 9.99], TEN, n=NP_T,
                   pmts=[0.05, 0.10], lz_barr=[0.50, np.nan], lz_pmt=[0.013, 0.0])
    oi = _obs_day(2)[0]
    assert abs(v - (1 + 0.013) * np.exp(-RATE * oi / 365)) < 1e-5


def test_lizard_absent_matches_no_lizard_call():
    a = mc_daily_t(SIG0, EYE, BETA, 0.5, [9.99, 9.99], TEN, n=NP_T, pmts=[0.05, 0.10],
                   lz_barr=[np.nan, np.nan], lz_pmt=[0.0, 0.0])
    b = mc_daily_t(SIG0, EYE, BETA, 0.5, [9.99, 9.99], TEN, n=NP_T, pmts=[0.05, 0.10])
    assert abs(a - b) < 1e-6


def test_regular_redemption_wins_over_lizard_at_same_observation():
    v = mc_daily_t(SIG0, EYE, BETA, 0.5, [0.90, 0.90], TEN, n=NP_T,
                   pmts=[0.05, 0.10], lz_barr=[0.50, np.nan], lz_pmt=[0.001, 0.0])
    oi = _obs_day(2)[0]
    assert abs(v - (1 + 0.05) * np.exp(-RATE * oi / 365)) < 1e-5     # 리자드 0.001 이 아니다


def test_lizard_does_not_trigger_when_barrier_breached():
    # 배리어를 1.2 로 올리면 발행 직후부터 no-touch 조건이 깨져 발동하지 않아야 한다.
    v = mc_daily_t(SIG0, EYE, BETA, 0.5, [9.99, 9.99], TEN, n=NP_T,
                   pmts=[0.05, 0.10], lz_barr=[1.20, np.nan], lz_pmt=[0.99, 0.0])
    assert abs(v - np.exp(-RATE)) < 1e-5                              # 만기 원금 1.0 (낙인 미터치)


def test_barrier_one_forces_knock_in():
    # 변동성 0 이면 worst = exp(0.02t) > 1 이라 B=1.0 을 '터치'하지 않는다.
    # 노낙인 규칙이므로 만기 미상환 경로는 worst 를 받아야 한다 -> 가격 1.0
    v_noki = mc_daily_t(SIG0, EYE, BETA, 1.0, [9.99, 9.99], TEN, n=NP_T, pmts=[0.0, 0.0])
    assert abs(v_noki - 1.0) < 1e-5
    v_ki = mc_daily_t(SIG0, EYE, BETA, 0.5, [9.99, 9.99], TEN, n=NP_T, pmts=[0.0, 0.0])
    assert abs(v_ki - np.exp(-RATE)) < 1e-5
    assert v_noki > v_ki


@pytest.mark.skipif(DEFAULT_DEV != "cuda", reason="CUDA 없음")
def test_cpu_and_cuda_agree_on_deterministic_case():
    a = mc_daily_t(SIG0, EYE, BETA, 0.5, [0.90, 0.90], TEN, n=NP_T, pmts=[0.02, 0.04], dev="cpu")
    b = mc_daily_t(SIG0, EYE, BETA, 0.5, [0.90, 0.90], TEN, n=NP_T, pmts=[0.02, 0.04], dev="cuda")
    assert abs(a - b) < 1e-6
