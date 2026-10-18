# -*- coding: utf-8 -*-
"""계약조건 섭동 — 시장상태는 그대로 두고 계약만 바꾼다 (MC 재실행 없음).

대리모형의 입력만 바꿔 출력을 보는 민감도 분석의 기반이다. 전부 순수 함수(CON 배열 in/out)라
학습·추론 어느 쪽에도 의존하지 않는다. `module/train.py` 의 단조손실 `_perturb` 는 같은 규칙의
torch 판이다 — 규칙을 바꾸면 **양쪽을 같이** 고쳐야 한다.

module/schedule.py 의 표현을 유지한다: 12칸 고정, 행사가·지급률은 마지막값 채움,
리자드 없으면 중립값(LZ_BARR_NEUTRAL=1.2 / LZ_PMT_NEUTRAL=0).

  쿠폰        누적지급률 pmt_j 를 같은 비율로 재조정 (스케줄 모양 유지)
  KI 배리어    B 만 변경 (B >= 1.0 은 노낙인 → 리자드도 중립으로)
  행사가 수준   스케줄 전체 평행이동
  스텝다운 폭   첫 행사가 고정, 첫−마지막 간격을 목표폭으로 비례 조정
  리자드       배리어 평행이동 · 지급률 비례 (리자드 상품만)

종전에는 scratch/contract_sensitivity.py 에만 있어 추적 파이프라인에서 쓸 수 없었다.
"""
import numpy as np

from .data import CONTRACT
from .schedule import LZ_BARR_NEUTRAL, LZ_PMT_NEUTRAL

iS, iP, iLB, iLP = slice(0, 12), slice(12, 24), slice(24, 36), slice(36, 48)
iB = CONTRACT.index("BARR_1/100")
iC = CONTRACT.index("ANL_RTRN/100")
iT = CONTRACT.index("TENOR")


def set_coupon(CON, c):
    """연쿠폰을 c 로. 누적지급률을 같은 비율로 재조정해 스케줄 모양을 유지한다.

    입력
      CON : ndarray(n, 51) — 계약 블록. 원본은 바꾸지 않는다
      c   : float — 목표 연쿠폰 (소수)
    출력
      ndarray(n, 51) — 사본. 기존 쿠폰이 0 이면 지급률도 0 으로 둔다 (비율을 알 수 없다)
    """
    X = CON.copy()
    old = X[:, iC]
    X[:, iP] = X[:, iP] * np.where(old > 1e-9, c / np.maximum(old, 1e-9), 0.0)[:, None]
    X[:, iC] = c
    return X


def set_barrier(CON, b):
    """KI 배리어를 b 로. b >= 1.0 은 노낙인이므로 리자드도 중립으로 되돌린다 (구조 일관성).

    입력
      CON : ndarray(n, 51) — 계약 블록
      b   : float — 목표 배리어 (소수, 스팟=1.0 기준)
    출력
      ndarray(n, 51) — 사본. b >= 1.0 이면 lz_barr 를 1.2, lz_pmt 를 0 으로 덮어쓴다
    """
    X = CON.copy()
    X[:, iB] = b
    if b >= 1.0:
        X[:, iLB] = LZ_BARR_NEUTRAL
        X[:, iLP] = LZ_PMT_NEUTRAL
    return X


def shift_strikes(CON, d):
    """행사가 스케줄 12칸 전체를 d 만큼 평행이동한다.

    입력
      CON : ndarray(n, 51) — 계약 블록
      d   : float — 이동폭 (소수). 음수면 행사가가 낮아진다
    출력
      ndarray(n, 51) — 사본. 스텝다운 폭(첫−마지막 간격)은 바뀌지 않는다
    """
    X = CON.copy()
    X[:, iS] = X[:, iS] + d
    return X


def set_spread(CON, sp, nobs):
    """첫 행사가를 고정하고 첫−마지막 간격을 sp 로 맞춘다.

    원래 평평한(간격 0) 스케줄은 비례 조정이 불가능하므로 회차 수에 맞춰 **선형 스텝다운을
    새로 만들고** 남는 칸은 마지막값으로 채운다.

    입력
      CON  : ndarray(n, 51) — 계약 블록
      sp   : float — 목표 스텝다운 폭 (소수)
      nobs : array-like(n,) int — 상품별 회차 수. 평평한 스케줄을 새로 만들 때만 쓴다
    출력
      ndarray(n, 51) — 사본
    """
    X = CON.copy()
    Sx = X[:, iS].copy()
    first = Sx[:, 0]
    old = first - Sx[:, -1]
    flat = old <= 1e-9
    r = np.where(flat, 0.0, sp / np.maximum(old, 1e-9))
    Sx = first[:, None] - (first[:, None] - Sx) * r[:, None]
    if flat.any():  # 평평하면 회차에 따라 균등 하락 후 마지막값 채움
        for i in np.where(flat)[0]:
            n = int(nobs[i])
            step = np.linspace(0, sp, n)
            Sx[i, :n] = first[i] - step
            Sx[i, n:] = Sx[i, n - 1]
    X[:, iS] = Sx
    return X


def shift_lizard(CON, d, has_lz):
    """리자드 배리어를 d 만큼 평행이동한다 (리자드가 켜진 회차만).

    입력
      CON    : ndarray(n, 51) — 계약 블록
      d      : float — 이동폭 (소수)
      has_lz : ndarray(n,) bool — 리자드 상품인가. False 행은 손대지 않는다
    출력
      ndarray(n, 51) — 사본. 중립값(1.2)인 회차는 그대로 둔다 — 그 회차엔 조항이 없다
    """
    X = CON.copy()
    lb = X[:, iLB]
    on = (lb < LZ_BARR_NEUTRAL) & has_lz[:, None]
    X[:, iLB] = np.where(on, lb + d, lb)
    return X


def scale_lizard_pmt(CON, r, has_lz):
    """리자드 지급률을 r 배로 한다.

    입력
      CON    : ndarray(n, 51) — 계약 블록
      r      : float — 배수
      has_lz : ndarray(n,) bool — 리자드 상품인가. False 행은 손대지 않는다
    출력
      ndarray(n, 51) — 사본. 중립값 0 은 몇 배를 해도 0 이라 자연히 보존된다
    """
    X = CON.copy()
    X[:, iLP] = np.where(has_lz[:, None], X[:, iLP] * r, X[:, iLP])
    return X


GRID = {
    "barrier": ("KI 배리어", np.round(np.arange(0.35, 0.661, 0.05), 3).tolist() + [1.0]),
    "coupon": ("쿠폰(연)", np.round(np.arange(0.03, 0.1001, 0.005), 4).tolist()),
    "strike_shift": ("행사가 평행이동", np.round(np.arange(-0.10, 0.0501, 0.025), 4).tolist()),
    "spread": ("스텝다운 폭", np.round(np.arange(0.0, 0.3501, 0.05), 3).tolist()),
    "lz_shift": ("리자드 배리어 이동", np.round(np.arange(-0.10, 0.1001, 0.05), 3).tolist()),
    "lz_pmt_scale": ("리자드 지급률 배수", [0.5, 0.75, 1.0, 1.25, 1.5]),
}


def apply_factor(CON, factor, lv, nobs, has_lz):
    """GRID 의 요인 이름 하나와 수준 하나로 계약을 변형한다 — 위 여섯 함수의 단일 입구.

    입력
      CON    : ndarray(n, 51) — 계약 블록
      factor : str — GRID 의 키 ("barrier" | "coupon" | "strike_shift" | "spread" |
               "lz_shift" | "lz_pmt_scale")
      lv     : float — 그 요인의 수준
      nobs   : array-like(n,) — "spread" 에만 쓰인다
      has_lz : ndarray(n,) bool — 리자드 두 요인에만 쓰인다
    출력
      ndarray(n, 51) — 변형된 사본
    예외
      ValueError — 모르는 factor
    """
    if factor == "barrier":
        return set_barrier(CON, lv)
    if factor == "coupon":
        return set_coupon(CON, lv)
    if factor == "strike_shift":
        return shift_strikes(CON, lv)
    if factor == "spread":
        return set_spread(CON, lv, nobs)
    if factor == "lz_shift":
        return shift_lizard(CON, lv, has_lz)
    if factor == "lz_pmt_scale":
        return scale_lizard_pmt(CON, lv, has_lz)
    raise ValueError(factor)


def in_domain(src, st, factor, lv, struct):
    """그 구조의 표본에서 해당 수준 근처(±허용)에 실제 상품이 몇 % 인가 — 외삽 표시용.

    0 에 가까우면 그 수준은 **학습 분포 밖**이고, 거기서 나온 기울기는 외삽이다.

    입력
      src    : DataFrame — 원천. B · coupon · Kfirst · stepdown 열을 본다
      st     : array-like(n,) — 행별 구조 라벨
      factor : str — "barrier" | "coupon" | "strike_shift" | "spread".
               리자드 두 요인은 지원하지 않는다
      lv     : float — 수준. "strike_shift" 는 이동폭이므로 그 구조의 Kfirst 중위수에 더해 쓴다
      struct : 비교할 구조 라벨
    출력
      float — 0~100 [%]. 지원하지 않는 factor 면 np.nan
    """
    m = st == struct
    if factor == "barrier":
        x, tol = src.B.values[m], 0.02
    elif factor == "coupon":
        x, tol = src.coupon.values[m], 0.0035
    elif factor == "strike_shift":
        x, tol = src.Kfirst.values[m], 0.02
    elif factor == "spread":
        x, tol = src.stepdown.values[m], 0.03
    else:
        return np.nan
    if factor == "strike_shift":
        lv = np.median(src.Kfirst.values[m]) + lv
    return float(np.mean(np.abs(x - lv) <= tol) * 100)
