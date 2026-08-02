# -*- coding: utf-8 -*-
"""완전 MC(numpy 레거시 엔진) vs V3(torch) 200상품 대조. (pytest 수집 대상 아님 — 수동 실행)

 두 엔진이 같은 규칙을 계산할 수 있는 구간에서만 비교한다:
   STEP x 낙인 상품 + 레거시 규칙(선형 쿠폰 c*t, 리자드 없음, 낙인 판정).
 이러면 규칙 차이가 제거되고 구현·난수 차이만 남는다.
 새 규칙(회차별 지급률·리자드·노낙인)의 정확성은 tests/test_mc_engine.py 의 sigma=0 결정론 테스트가 담당한다.

 판정: 두 독립 MC 추정치 차이의 표준편차는 sqrt(2)*SE 이므로 z=(v_t - v_np)/(sqrt(2)*SE) 가
       표준정규처럼 흩어져야 한다. SE 는 보수적 상한 0.0015 를 쓴다(노트북 4 측정 0.0002~0.0019).

 실행: python tests/verify_engine_agreement.py [n_products]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from util import file_manager as fm
from util import plot as _plot          # 학술 rcParams
import matplotlib.pyplot as plt
from module import mc as MC
from module.mc_engine import mc_daily_t, DEFAULT_DEV

NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SE = 0.0015
PASS_MEAN_Z, PASS_TAIL, PASS_MAE = 0.3, 0.02, 0.003


def main():
    df = pd.read_parquet(fm.source()).sort_values("isu_ord").reset_index(drop=True)
    pool = df[(df["opt_type"] == "STEP") & (df["ki_yn"] == 1)]
    assert len(pool) >= NSAMP, f"STEP x 낙인 상품이 {len(pool)}개뿐"
    idx = np.random.default_rng(0).choice(pool.index.values, NSAMP, replace=False)
    mk = MC.load_market()
    print(f"대조 {NSAMP}건 (STEP x 낙인, 레거시 규칙) | torch device {DEFAULT_DEV}", flush=True)

    rows = []
    t0 = time.time()
    for c_, i in enumerate(idx):
        r = df.loc[i]
        strikes = [float(r[f"strk_{j}"]) for j in range(int(r["nobs"]))]
        P = MC.prepare_one(mk, None, r["item"], int(r["isu_ord"]), float(r["tenor"]),
                           float(r["sig_eff"]), strikes=strikes)
        a = (P["sigs"], P["corr"], P["beta"], float(r["B"]), P["strikes"])
        v_np = MC.mc_daily(*a, float(r["coupon"]), float(r["tenor"]), seed=int(i))
        v_t = mc_daily_t(*a, float(r["tenor"]), c=float(r["coupon"]), seed=int(i))
        rows.append(dict(i=int(i), ten=float(r["tenor"]), numpy=v_np, torch=v_t,
                         d=v_t - v_np, z=(v_t - v_np) / (np.sqrt(2) * SE)))
        if (c_ + 1) % 25 == 0:
            print(f"  {c_+1}/{NSAMP} ({time.time()-t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)

    mean_z = float(R.z.mean()); tail = float((R.z.abs() > 3).mean()); mae = float(R.d.abs().mean())
    ok = (abs(mean_z) < PASS_MEAN_Z) and (tail < PASS_TAIL) and (mae < PASS_MAE)
    print(f"\nmean z   {mean_z:+.3f}   (기준 |z| < {PASS_MEAN_Z})")
    print(f"sd z     {R.z.std(ddof=1):.3f}   (기대 ~1.0)")
    print(f"|z|>3    {tail:.1%}      (기준 < {PASS_TAIL:.0%})")
    print(f"MAE      {mae:.5f}   ({mae*10000:.1f}원/액면1만, 기준 < {PASS_MAE})")
    print(f"|d|max   {R.d.abs().max():.5f}")
    print(f"\n판정: {'PASS — V3 를 정본 엔진으로 사용해도 된다' if ok else 'FAIL — mc_engine.py 를 점검하라'}")

    fig, ax = plt.subplots(1, 2, figsize=(13.333, 4.6))
    ax[0].scatter(np.arange(len(R)), R.z, s=24, color="#1f6f9c", edgecolors="white", linewidths=0.4)
    for y in (-3, 3):
        ax[0].axhline(y, color="#c0392b", lw=1.1, ls="--")
    ax[0].axhline(0, color="#888888", lw=0.8)
    ax[0].set_ylim(-4.2, 4.2); ax[0].set_xlabel("product")
    ax[0].set_ylabel(r"$z=(v_{V3}-v_{MC})/(\sqrt{2}\,SE)$")
    ax[0].set_title(f"Engine agreement  (mean z = {mean_z:+.2f})", fontsize=12)
    ax[0].grid(axis="y", color="#dddddd", lw=0.5, alpha=0.6)

    lim = [min(R.numpy.min(), R.torch.min()), max(R.numpy.max(), R.torch.max())]
    ax[1].scatter(R.numpy, R.torch, s=18, alpha=0.5, color="#1f6f9c", edgecolors="none")
    ax[1].plot(lim, lim, "--", color="#c0392b", lw=1.1)
    ax[1].set_xlabel("full MC price (numpy)"); ax[1].set_ylabel("V3 price (torch)")
    ax[1].set_title(f"MAE {mae:.5f}  ({mae*10000:.0f} KRW / 10,000 face)", fontsize=12)
    ax[1].grid(color="#dddddd", lw=0.5, alpha=0.6)

    fig.suptitle(f"Full MC vs V3 engine - {NSAMP} products (STEP x knock-in, legacy rules)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fm.image("mc_engine_agreement"), dpi=400, bbox_inches="tight", transparent=True)
    print("saved ->", fm.image("mc_engine_agreement").name)
    R.to_csv(fm.stat("engine_agreement"), index=False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
