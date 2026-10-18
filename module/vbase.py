# -*- coding: utf-8 -*-
"""논문의 준해석적 베이스 프록시 V_base 를 **우리 실제 ELS** 에 이식 (식 19).

우리 1단계 쿼리는 전부 발행일(t0=0, w0=1, q0=0) 이므로 식이 단순해진다.
  V_base = sum_j (C_{j-1} - C_j)(1 + pmt_j) DF(theta_j)
           + C_K DF(T) [ p_KI * E[w_T | w_T < b_K] + (1 - p_KI) * 1.0 ]

원문과 달라지는 두 곳 (우리 MC 의 실제 페이오프에 맞춤)
  ① 만기 비상환·미낙인 분기: 원문은 1+cT, 우리 MC 는 **원금 1.0** (mc_daily_t 의 sv 분기).
  ② 만기 손실 분기: 원문은 손실 페이오프를 w_0 로 대체하는데, 우리는 t0=0 이라 w_0=1 이
     되어 KI 항이 통째로 사라진다. 그래서 스칼라 GBM 의 **절단 로그정규 평균**
     E[w_T | w_T < b_K] 로 바꾼다 (여전히 닫힌 형태, 시뮬레이션 없음).
  ③ B >= 1.0 은 '항상 낙인'(always_ki) 이므로 p_KI = 1.

그대로 두는 원문의 단순화: worst-name 교체효과 · 완전한 상관효과 · 조기상환/낙인/만기 의존성 무시.
리자드(조기상환 완화 조항)도 프록시에서는 무시한다 — 망이 보정할 몫이다.

종전에는 scratch/els_vbase.py 에만 있어 추적 파이프라인에서 쓸 수 없었다.
프록시 단독 성능 리포트는 generate/report_vbase.py 가 만든다.
"""
import numpy as np
import torch

from .features import MSAMP

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NS = 12  # 계약 스케줄 슬롯 수
iS, iP, iLB, iLP = slice(0, NS), slice(NS, 2 * NS), slice(2 * NS, 3 * NS), slice(3 * NS, 4 * NS)
iB, iC, iT = 4 * NS, 4 * NS + 1, 4 * NS + 2
_G = None


def _grid(dev):
    global _G
    if _G is None or _G[0].device.type != dev:
        G, R = 801, 20.0
        x = torch.linspace(-R, R, G, device=dev)
        dx = float(x[1] - x[0])
        K = torch.exp(-0.5 * (x[None, :] - x[:, None]) ** 2) / np.sqrt(2 * np.pi) * dx
        _G = (x, K, dx)
    return _G


def _clark_min(m, v, rho):
    """Clark(1961) 근사로 min(Y_1,Y_2,Y_3) 의 평균·분산. m,v: (n,3), rho: (n,) 공통상관.
    min = -max(-Y) 를 쓴다. 논문의 sigma_bar 는 worst-of 효과를 통째로 무시하므로,
    같은 '스칼라 1개로 대체' 라는 틀 안에서 분포의 1·2차 적률만 맞춰 준다."""
    nd = torch.special.ndtr
    phi = lambda x: torch.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    mm, vv = -m, v
    m1, v1, m2, v2 = mm[:, 0], vv[:, 0], mm[:, 1], vv[:, 1]
    a = torch.sqrt((v1 + v2 - 2 * rho * torch.sqrt(v1 * v2)).clamp(min=1e-12))
    al = (m1 - m2) / a
    E1 = m1 * nd(al) + m2 * nd(-al) + a * phi(al)
    E2 = (v1 + m1**2) * nd(al) + (v2 + m2**2) * nd(-al) + (m1 + m2) * a * phi(al)
    V1 = (E2 - E1**2).clamp(min=1e-12)
    r13 = (torch.sqrt(v1) * rho * nd(al) + torch.sqrt(v2) * rho * nd(-al)) / torch.sqrt(V1)
    m3, v3 = mm[:, 2], vv[:, 2]
    a2 = torch.sqrt((V1 + v3 - 2 * r13 * torch.sqrt(V1 * v3)).clamp(min=1e-12))
    al2 = (E1 - m3) / a2
    Em = E1 * nd(al2) + m3 * nd(-al2) + a2 * phi(al2)
    Em2 = (V1 + E1**2) * nd(al2) + (v3 + m3**2) * nd(-al2) + (E1 + m3) * a2 * phi(al2)
    return -Em, (Em2 - Em**2).clamp(min=1e-12)


def _scalarize(V, nstar, dev):
    """자산 수에 맞는 worst-of 스칼라화용 (sigma 목록, 공통 rho). 패딩 중복을 풀어낸다.

    VC 는 sigma 를 정렬해 담으므로 2스타 패딩(A,B,B)은 중복값이 두 번 나타난다.
    중복값과 나머지를 갈라 **실제 자산 수만큼만** Clark 을 돌린다.
    1스타는 Clark 자체가 불필요하다 (worst-of = 그 자산).
    """
    s = V[:, :3]
    r3 = V[:, 3:6]
    n = len(s)
    if nstar is None:
        return s, r3.mean(1).clamp(-0.99, 0.99), torch.full((n,), 3, device=dev)
    ns = torch.as_tensor(np.asarray(nstar), dtype=torch.int64, device=dev)
    # 중복값 판정: 정렬돼 있으므로 인접쌍만 보면 된다
    d01 = (s[:, 1] - s[:, 0]).abs() < 1e-9
    dup = torch.where(d01, s[:, 0], s[:, 2])  # 두 번 나타난 값
    oth = torch.where(d01, s[:, 2], s[:, 0])  # 한 번만 나타난 값
    s2 = torch.stack([dup, oth, oth], 1)  # 2스타: 앞 두 칸만 쓴다
    rho2 = r3.min(1).values.clamp(-0.99, 0.99)  # 중복쌍 1.0 을 뺀 실제 상관
    s1 = s[:, :1].expand(-1, 3)
    sig = torch.where((ns == 1)[:, None], s1, torch.where((ns == 2)[:, None], s2, s))
    rho = torch.where(ns == 2, rho2, r3.mean(1).clamp(-0.99, 0.99))
    return sig, rho, ns


@torch.no_grad()
def vbase_els(CON, VC, CURVE, nobs, dev=DEV, mode="mean", lizard=False, nstar=None):
    """우리 stage-1 입력 -> V_base (액면 1 단위). CON/VC/CURVE 는 D.CON/D.VC/D.CURVE.

    mode: 3자산을 스칼라 하나로 줄이는 방식
      "mean"  sigma_bar = 평균 변동성 (원문 그대로)
      "eff"   sig_eff  = 우리 파이프라인의 바스켓 실효변동성 (VC 의 7번째 열)
      "clark" 만기 log-worst-of 의 평균·분산을 Clark 근사로 맞춘 스칼라 (worst-of 효과 반영)
    """
    T_ = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device=dev)
    C, V, U = T_(CON), T_(VC), T_(CURVE)
    K = T_(nobs, torch.int64)
    n = len(C)
    ten = C[:, iT]
    sb = V[:, :3].mean(1)  # sigma_bar
    qb = V[:, 7:10].mean(1) if V.shape[1] >= 10 else torch.zeros(n, device=dev)
    B = C[:, iB]
    idx = torch.arange(NS, device=dev)[None, :]
    on = idx < K[:, None]  # 유효 관측일
    th = ten[:, None] * (idx + 1).to(torch.float32) / K[:, None].to(torch.float32)  # theta_k
    bb = C[:, iS]  # 행사가(배리어) 스케줄
    pm = C[:, iP]  # 회차별 누적 지급률

    # --- 할인·드리프트: 부트스트랩 제로금리 곡선에서 읽는다 ---
    ms = T_(MSAMP)

    def zrate(t):  # (n,NS) 선형보간 + 평탄외삽
        tt = t.clamp(min=1e-6)
        j = torch.searchsorted(ms, tt.contiguous()).clamp(1, len(ms) - 1)
        t0_, t1_ = ms[j - 1], ms[j]
        y0 = torch.gather(U, 1, (j - 1).clamp(max=U.shape[1] - 1))
        y1 = torch.gather(U, 1, j.clamp(max=U.shape[1] - 1))
        w = ((tt - t0_) / (t1_ - t0_)).clamp(0, 1)
        return y0 + (y1 - y0) * w

    z = zrate(th)  # (n,NS) 만기별 제로금리
    DF = torch.exp(-z * th)
    zT = zrate(ten[:, None])[:, 0]
    if mode == "eff":
        sb = V[:, 6]
        M = (z - qb[:, None]) * th - 0.5 * sb[:, None] ** 2 * th
    elif mode == "clark":
        sig3, rho, ns = _scalarize(V, nstar, dev)
        q3 = V[:, 7:10]
        mA = (zT[:, None] - q3) * ten[:, None] - 0.5 * sig3**2 * ten[:, None]
        vA = sig3**2 * ten[:, None]
        Mw, Vw = _clark_min(mA, vA, rho)  # 만기 log-worst-of 의 적률 (3자산)
        if nstar is not None:
            # 1스타: worst-of = 그 자산. 2스타: 앞 두 칸만으로 Clark 1단계.
            m1, v1 = mA[:, 0], vA[:, 0]
            a2 = torch.sqrt(
                (vA[:, 0] + vA[:, 1] - 2 * rho * torch.sqrt(vA[:, 0] * vA[:, 1])).clamp(min=1e-12)
            )
            al = (-mA[:, 0] + mA[:, 1]) / a2
            nd0 = torch.special.ndtr
            ph = lambda x: torch.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            Em = -mA[:, 0] * nd0(al) - mA[:, 1] * nd0(-al) + a2 * ph(al)
            Em2 = (
                (vA[:, 0] + mA[:, 0] ** 2) * nd0(al)
                + (vA[:, 1] + mA[:, 1] ** 2) * nd0(-al)
                + (-mA[:, 0] - mA[:, 1]) * a2 * ph(al)
            )
            Mw = torch.where(ns == 1, m1, torch.where(ns == 2, -Em, Mw))
            Vw = torch.where(ns == 1, v1, torch.where(ns == 2, (Em2 - Em**2).clamp(min=1e-12), Vw))
        sb = torch.sqrt(Vw / ten.clamp(min=1e-8))
        M = (Mw / ten.clamp(min=1e-8))[:, None] * th  # 드리프트는 시간에 선형
    else:  # "mean" — 원문
        M = (z - qb[:, None]) * th - 0.5 * sb[:, None] ** 2 * th

    # --- C_j^surv : 무드리프트 랜덤워크 순차 합성곱 (관측일 등간격) ---
    x, Kr, dx = _grid(dev)
    G = len(x)
    Dl = (ten / K.to(torch.float32)).clamp(min=1e-8)  # 관측 간격
    s_unit = (sb * torch.sqrt(Dl)).clamp(min=1e-12)
    A = (torch.log(bb.clamp(min=1e-6)) - M) / s_unit[:, None]
    p = torch.zeros(n, G, device=dev)
    Cs = torch.ones(n, NS + 1, device=dev)
    first = torch.ones(n, dtype=torch.bool, device=dev)
    for j in range(NS):
        o = on[:, j]
        Cs[:, j + 1] = Cs[:, j]
        if not bool(o.any()):
            continue
        pn = torch.where(
            first[:, None],
            torch.exp(-0.5 * x[None, :] ** 2) / np.sqrt(2 * np.pi) * torch.ones(n, 1, device=dev),
            p @ Kr,
        )
        p = torch.where(o[:, None], pn * (x[None, :] < A[:, j][:, None]).to(p.dtype), p)
        Cs[:, j + 1] = torch.where(o, (p.sum(1) * dx).clamp(0, 1), Cs[:, j])
        first = first & ~o

    # --- p_KI : 연속배리어 최초통과 (식 17-18), B>=1 은 항상 낙인 ---
    mT_ = torch.gather(M, 1, (K - 1)[:, None]).squeeze(1)
    nu = mT_ / ten.clamp(min=1e-8)  # 스칼라 과정의 로그드리프트율
    sq = (sb * torch.sqrt(ten)).clamp(min=1e-12)
    bl = torch.log(B.clamp(min=1e-6))
    nd = type("N", (), {"cdf": staticmethod(torch.special.ndtr)})()  # NaN 행 허용 (제외행)
    pki = nd.cdf((bl - nu * ten) / sq) + torch.exp((2 * nu * bl / sb**2).clamp(max=50)) * nd.cdf(
        (bl + nu * ten) / sq
    )
    pki = torch.where(B >= 0.999, torch.ones_like(pki), pki.clamp(0, 1))

    # --- 만기 손실분기: E[w_T | w_T < b_K] (절단 로그정규) ---
    bK = torch.gather(bb, 1, (K - 1)[:, None]).squeeze(1)
    mT = mT_
    sT = sq
    d_ = (torch.log(bK.clamp(min=1e-6)) - mT) / sT
    Pl = nd.cdf(d_).clamp(min=1e-9)
    Ew = torch.exp(mT + 0.5 * sT**2) * nd.cdf(d_ - sT) / Pl  # 조건부 평균

    # --- 리자드(선택): "누적최저가 >= 리자드배리어" 는 최초통과 확률의 닫힌 형태 ---
    #  X_t = ln w~_t = M(t) + sigma_bar * W_t 에 대해
    #  P(min_{[0,T]} X >= l, X_T < a)
    #    = Phi((a-MT)/s) - Phi((l-MT)/s) - exp(2*mu*l/sig^2)[Phi((a-2l-MT)/s) - Phi((-l-MT)/s)]
    #  정규 조기상환(X_T >= a)과 배타적이므로 같은 회차에서 겹치지 않는다.
    nd_ = torch.special.ndtr
    if lizard:
        lzb, lzp = C[:, iLB], C[:, iLP]
        act = (lzb < 1.2) & on
        sdj = (sb[:, None] * torch.sqrt(th)).clamp(min=1e-12)
        mu_j = M / th.clamp(min=1e-8)
        el = torch.log(lzb.clamp(min=1e-6))
        aa = torch.log(bb.clamp(min=1e-6))
        Pj = (
            nd_((aa - M) / sdj)
            - nd_((el - M) / sdj)
            - torch.exp((2 * mu_j * el / sb[:, None] ** 2).clamp(max=50))
            * (nd_((aa - 2 * el - M) / sdj) - nd_((-el - M) / sdj))
        )
        Pj = torch.where(act, Pj.clamp(0, 1), torch.zeros_like(Pj))
    else:
        lzp = torch.zeros_like(bb)
        Pj = torch.zeros_like(bb)

    # --- 식 (19) : 회차별로 생존을 갱신하며 적립 ---
    Sv = torch.ones(n, device=dev)  # 아직 상환되지 않은 확률
    v = torch.zeros(n, device=dev)
    for j in range(NS):
        o = on[:, j]
        Cprev = Cs[:, j].clamp(min=1e-9)
        q_reg = ((Cs[:, j] - Cs[:, j + 1]) / Cprev).clamp(0, 1)  # 조건부 정규상환
        # 리자드 적격집합은 '정규 미상환 생존집합'의 부분집합이어야 한다 -> C_j 로 상한
        joint = torch.minimum(Pj[:, j], Cs[:, j + 1])
        q_lz = (joint / Cprev).clamp(min=0)
        q_lz = torch.minimum(q_lz, (1.0 - q_reg).clamp(min=0))  # 합이 1 을 넘지 않게
        pay = q_reg * (1.0 + pm[:, j]) + q_lz * (1.0 + lzp[:, j])
        v = v + torch.where(o, Sv * pay * DF[:, j], torch.zeros_like(v))
        Sv = torch.where(o, Sv * (1.0 - q_reg - q_lz).clamp(min=0), Sv)
    DFT = torch.gather(DF, 1, (K - 1)[:, None]).squeeze(1)
    v = v + Sv * DFT * (pki * Ew + (1.0 - pki) * 1.0)
    return v.cpu().numpy()
