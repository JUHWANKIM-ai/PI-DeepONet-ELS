# -*- coding: utf-8 -*-
"""계약조건 반응을 OOS 전수 횡단면으로 — 기준상품 12개(리자드 6개) 섭동격자를 대체한다.

 이미 있는 MC 가격을 그대로 쓴다(MC 재계산 없음). 상품마다 자기 계약조건 값이 있으므로
   x = 그 상품의 계약조건,  y = MC 가격 / 모델 가격
 을 전부 점으로 찍고 각각 OLS 회귀선을 긋는다. 모델과 MC 가 **같은 상품** 위에
 올라가므로 교란(변동성·만기·발행시점)이 양쪽에 동일하게 걸려 두 선의 비교는 유효하다.

 핵심 지표는 잔차회귀다:  (모델 − MC) ~ 계약조건.
 기울기가 0 이면 모델이 그 조건에 MC 와 똑같이 반응한다는 뜻이고, 공통 교란은 차분에서 상쇄된다.
 표준오차는 발행일 군집(같은 날 발행 상품은 시장입력을 공유)으로 보정한다.

 주의: 횡단면 기울기는 편미분이 아니라 연관이다. KI 배리어는 횡단면 기울기가 +인데
 섭동 편미분은 −다(배리어 깊은 상품이 쿠폰·변동성도 다름). 절대값을 민감도로 읽지 말고
 MC 선 대비 모델 선의 차이만 읽는다.

 자료는 1·2·3스타 통합(stage1_all_stars)이고 모형도 전부 통합 학습본이라 스타별로 나눠 낸다.

 산출: result/statistics/xsec_slope.csv · xsec_resid.csv  (nstar 열 포함, ALL 행 포함)
 실행: python tools/slope_cross_section.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module import contract as CS
from module import anchors as AN
from module import star as A
from module.schedule import LZ_BARR_NEUTRAL
from util import file_manager as fm

FACE = 10000
SEEDS = list(AN.SEEDS)
MIN_N = 30                      # 이보다 적으면 회귀를 내지 않는다 (1·2스타 리자드가 얇다)
EN = {"barrier": "KI barrier", "coupon": "coupon (annual)", "strike_shift": "first strike",
      "spread": "step-down width", "lz_shift": "lizard barrier",
      "lz_pmt_scale": "lizard payout"}
# (표시이름, 접두사, 잔차타깃, V_base 피처, 종류)
# 기준 모형은 Direct + 단조손실. 학습 풀만 바꾼 두 벌을 같은 폴드·같은 시험집합에서 비교하고,
# XGBoost(통합 학습)를 벤치마크로 둔다. 프록시(Residual)는 이 그림에 넣지 않는다.
M = [("Direct+m (1·2·3)", "ball_d", False, False, "don"),
     ("Direct+m (3star only)", "ball3_d", False, False, "don"),
     ("XGBoost (1·2·3)", "ball_xgb", False, False, "xgb")]

_c = {}


def pred(pre, kind, s, k):
    key = (pre, s, k)
    if key not in _c:
        if kind == "xgb":
            _c[key] = AN.load_xgb(str(AN.MDIR / "{}_anchor_seed{}_fold{}".format(pre, s, k)) + ".pkl")
        else:
            _c[key] = AN.load_don(
                str(AN.MDIR / "{}_deeponet_anchor_seed{}_fold{}".format(pre, s, k)) + ".pt")
    return _c[key]


def ols_cluster(x, y, g):
    """y = a + b x, 군집(g) 보정 표준오차. 반환 (b, se_b, a, se_a, R2, n, n_clu)."""
    X = np.c_[np.ones(len(x)), x]
    bet = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ bet
    XtXi = np.linalg.inv(X.T @ X)
    uniq, inv = np.unique(g, return_inverse=True)
    S = np.zeros((2, 2))
    for c in range(len(uniq)):                       # sum_c (X_c' e_c)(X_c' e_c)'
        m = inv == c
        s = X[m].T @ e[m]
        S += np.outer(s, s)
    G, n = len(uniq), len(x)
    adj = G / (G - 1) * (n - 1) / (n - 2)
    V = XtXi @ (adj * S) @ XtXi
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return bet[1], np.sqrt(V[1, 1]), bet[0], np.sqrt(V[0, 0]), r2, n, G


def build_terms(D):
    """상품별 실제 계약조건 값. 리자드가 없는 상품은 리자드 항목이 NaN 이다."""
    nobs = np.asarray(D.nobs).astype(int)
    K = D.CON[:, CS.iS]
    on = D.CON[:, CS.iLB] < LZ_BARR_NEUTRAL
    sd = np.array([K[i, 0] - K[i, nobs[i] - 1] for i in range(len(D.CON))])
    with np.errstate(invalid="ignore"):
        lzb = np.nanmin(np.where(on, D.CON[:, CS.iLB], np.nan), 1)
        lzp = np.nanmax(np.where(on, D.CON[:, CS.iLP], np.nan), 1)
    return {"barrier": D.CON[:, CS.iB], "coupon": D.CON[:, CS.iC], "strike_shift": K[:, 0],
            "spread": sd, "lz_shift": lzb, "lz_pmt_scale": lzp}


def predictions(D):
    """실제 계약 그대로의 폴드별·5시드 평균 예측 (원 단위)."""
    vb = A.vbase_all(D)
    CONf = np.concatenate([D.CON, vb[:, None]], 1).astype("float32")
    fold = np.full(D.n, -1, int)
    for k, (tr, va, te) in enumerate(D.WF):
        fold[te] = k
    P = {lab: np.full(D.n, np.nan) for lab, *_ in M}
    for k in range(4):
        i = np.where(fold == k)[0]
        for lab, pre, px, ft, kind in M:
            X = (CONf if ft else D.CON)[i].astype("float32")
            v = np.mean([pred(pre, kind, s, k)(D.CURVE[i], D.VC[i], X) for s in SEEDS], 0)
            P[lab][i] = (v + (vb[i] if px else 0.0)) * FACE
    return P, fold


def groups(D, fold):
    oos = np.where(fold >= 0)[0]
    return [(1, oos[D.nstar[oos] == 1]), (2, oos[D.nstar[oos] == 2]),
            (3, oos[D.nstar[oos] == 3]), ("ALL", oos)]


def main():
    """OOS 전수 횡단면 기울기를 재어 xsec_slope.csv · xsec_resid.csv 를 만든다."""
    t0 = time.time()
    D = A.make_all()
    ref = build_terms(D)
    clu = np.asarray(D.ORD).astype(int)
    P, fold = predictions(D)
    y_mc = D.MC * FACE
    print("예측 완료 ({:.0f}s)".format(time.time() - t0), flush=True)

    lv, rv = [], []
    for ns, idx0 in groups(D, fold):
        for f, lab_f in EN.items():
            x0 = ref[f]
            idx = idx0[np.isfinite(x0[idx0]) & np.isfinite(y_mc[idx0])]
            if len(idx) < MIN_N or len(np.unique(x0[idx])) < 2:
                continue
            for name, y in [("MC (truth)", y_mc)] + [(l, P[l]) for l, *_ in M]:
                b, se, a, sea, r2, n, G = ols_cluster(x0[idx], y[idx], clu[idx])
                lv.append({"nstar": ns, "factor": lab_f, "series": name, "n": n, "n_clu": G,
                           "slope": b, "se": se, "intercept": a, "R2": r2})
            smc = lv[-len(M) - 1]["slope"]
            for lab, *_ in M:
                d = P[lab] - y_mc
                b, se, a, sea, r2, n, G = ols_cluster(x0[idx], d[idx], clu[idx])
                rv.append({"nstar": ns, "factor": lab_f, "model": lab, "n": n,
                           "MC_slope": smc, "resid_slope": b, "se": se, "t": b / se,
                           "rel": b / smc if abs(smc) > 1e-9 else np.nan,
                           "lo": b - 1.96 * se, "hi": b + 1.96 * se})
    S = pd.DataFrame(lv); S.to_csv(fm.stat("xsec_slope"), index=False, encoding="utf-8-sig")
    R = pd.DataFrame(rv); R.to_csv(fm.stat("xsec_resid"), index=False, encoding="utf-8-sig")

    for ns, _ in groups(D, fold):
        z = R[R.nstar == ns]
        if not len(z):
            continue
        print()
        print("=== {} | 잔차회귀 (모델 − MC) ~ 계약조건.  0 이면 MC 와 동일 반응 ===".format(
            "{}스타".format(ns) if ns != "ALL" else "전체"))
        for lab_f in z.factor.unique():
            w = z[z.factor == lab_f]
            print("  {} (n={:,}, MC 기울기 {:+.1f})".format(
                lab_f, int(w.n.iloc[0]), w.MC_slope.iloc[0]))
            for _, r in w.iterrows():
                print("     {:24} {:+9.1f} ± {:6.1f}  (MC 대비 {:+7.1%})  t {:+6.1f}{}".format(
                    r.model, r.resid_slope, r.se, r.rel, r.t,
                    "  <- 유의" if abs(r.t) >= 1.96 else ""))
    print()
    print("완료 {:.1f}분".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
