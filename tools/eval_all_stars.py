# -*- coding: utf-8 -*-
"""1·2·3스타 합산 stage-1 평가 — 스타별 수준 지표 + **스타별 계약조건 기울기**.

 모형 (전부 통합 · V_base 를 입력 피처로 사용 · 단조손실 lz_pmt margin 8원)
   ball_fm   Residual 타깃, 1·2·3스타로 학습
   ball3_fm  Residual 타깃, 3스타만으로 학습 (같은 폴드·같은 시험집합)
   ball_dm   Direct 타깃,   1·2·3스타로 학습
 기울기 참값은 tools/mc_check_12star.py 가 만든 스타별 MC 섭동 격자를 쓴다.

 산출: result/statistics/allstar_level.csv · allstar_slope.csv · result/image/allstar.png
 실행: python tools/eval_all_stars.py
"""
import sys, time
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module import star as A
from module import contract as CS
from module import anchors as AN
from module import market as MK
from module.vbase import vbase_els
from tools.mc_check_12star import con_of
from module.data import CONTRACT
from module.schedule import LZ_BARR_NEUTRAL
from util import file_manager as fm, plot as _plot

FACE = 10000
SEEDS = range(5)
UNIT = {"barrier": 0.1, "coupon": 0.01, "strike_shift": 0.1, "spread": 0.1,
        "lz_shift": 0.1, "lz_pmt_scale": 1.0}
EN = {"barrier": "KI barrier", "coupon": "coupon (annual)", "strike_shift": "strike level shift",
      "spread": "step-down width", "lz_shift": "lizard barrier shift",
      "lz_pmt_scale": "lizard payout multiple"}
# (표시이름, 접두사, 잔차타깃, V_base 피처, 색)  — 접두사가 _xgb 로 끝나면 XGBoost 체크포인트
M = [("Direct+m (1·2·3)", "ball_d", False, False, "#1f6f9c"),
     ("Direct+m (3star only)", "ball3_d", False, False, "#9b5de5"),
     ("XGBoost (1·2·3)", "ball_xgb", False, False, "#8d99ae"),
     ("XGBoost (3star only)", "ball3_xgb", False, False, "#c0c6d0"),
     ("Resid+f+m (1·2·3)", "ball_fm", True, True, "#2a9d8f"),
     ("Resid+f+m (3star only)", "ball3_fm", True, True, "#b07cc6"),
     ("Direct+f+m (1·2·3)", "ball_dm", False, True, "#f0c05a")]
# 헤드라인 그림: 3스타에서 기준 모형과 표형 벤치마크 둘만. 패널은 R²·MAPE 두 개로 고정한다
# (MAE·RMSE 등 나머지 지표는 allstar_level.csv 로만 보고한다).
HEADLINE = ("Direct+f+m (1·2·3)", "XGBoost (1·2·3)")
HEADLINE_PANELS = ("R2", "MAPE")

SHORT = ["Direct+m\n(trained 1·2·3)", "Direct+m\n(trained 3★ only)",
         "XGBoost\n(trained 1·2·3)", "XGBoost\n(trained 3★ only)",
         "Resid+f+m\n(trained 1·2·3)", "Resid+f+m\n(trained 3★ only)",
         "Direct+f+m\n(trained 1·2·3)"]


def met(p, y):
    e = p - y
    return dict(R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                MAPE=(np.abs(e) / np.abs(y)).mean() * 100, MAE=np.abs(e).mean(),
                RMSE=float(np.sqrt((e ** 2).mean())), MSE=float((e ** 2).mean()), bias=e.mean())


def main():
    """7종 모형을 스타별로 평가해 allstar_level/slope/slope_raw.csv 와 allstar.png 를 만든다."""
    t0 = time.time()
    D = A.make_all(); vb = A.vbase_all(D)
    CONf = np.concatenate([D.CON, vb[:, None]], 1).astype("float32")
    oos = np.concatenate([te for _, _, te in D.WF])
    fold = np.full(D.n, -1, int)
    for k, (tr, va, te) in enumerate(D.WF):
        fold[te] = k
    _dc = {}

    def don(p, s, k):                                  # 1,500행 격자에서 매번 load 하면 느리다
        key = (p, s, k)
        if key not in _dc:
            if p.endswith("_xgb"):
                _dc[key] = AN.load_xgb(
                    str(AN.MDIR / "{}_anchor_seed{}_fold{}".format(p, s, k)) + ".pkl")
            else:
                _dc[key] = AN.load_don(
                    str(AN.MDIR / "{}_deeponet_anchor_seed{}_fold{}".format(p, s, k)) + ".pt")
        return _dc[key]
    pr = {lab: np.full((5, D.n), np.nan) for lab, *_ in M}
    for k in range(4):
        i = np.where(fold == k)[0]
        for s in SEEDS:
            for lab, pre, res, ft, _ in M:
                X = CONf[i] if ft else D.CON[i]
                pr[lab][s, i] = don(pre, s, k)(D.CURVE[i], D.VC[i], X) + (vb[i] if res else 0)

    GR = [("ALL", np.ones(D.n, bool)), ("1star", D.nstar == 1),
          ("2star", D.nstar == 2), ("3star", D.nstar == 3)]
    rows = []
    for gl, gm in GR:
        idx = oos[gm[oos]]; y = D.MC[idx] * FACE
        for lab, *_ in M:
            per = [met(pr[lab][s, idx] * FACE, y) for s in SEEDS]
            r = {"group": gl, "model": lab, "n": len(idx), "mc_sd": y.std()}
            for k_ in per[0]:
                v = [p[k_] for p in per]
                r[k_] = float(np.mean(v)); r[k_ + "_se"] = float(np.std(v, ddof=1) / np.sqrt(len(v)))
            rows.append(r)
    T = pd.DataFrame(rows).set_index(["group", "model"])
    T.to_csv(fm.stat("allstar_level"), encoding="utf-8-sig")
    for gl, _ in GR:
        print()
        print("=== {} (n={:,}, MC sd {:.0f}원) ===".format(
            gl, int(T.loc[gl].n.iloc[0]), T.loc[gl].mc_sd.iloc[0]))
        for lab, *_ in M:
            r = T.loc[(gl, lab)]
            print("  {:26} R2 {:.4f}+-{:.4f} | MAPE {:.3f} | MAE {:5.1f}원 | 편향 {:+6.1f}원".format(
                lab, r.R2, r.R2_se, r.MAPE, r.MAE, r.bias))

    # ---------- 스타별 기울기 ----------
    C = pd.read_csv(fm.stat("mc_check_allstar"), encoding="utf-8-sig")
    d12 = pd.read_parquet(fm.DATA / "els12_dataset.parquet").sort_values("isu_ord").reset_index(drop=True)
    src = pd.concat([d12.assign(_s=d12.nstar), MK.dataset().assign(_s=3)], ignore_index=True).set_index("item")
    ord_of = dict(zip(D.ITEM, D.ORD)); fold_of = dict(zip(D.ITEM, fold))
    idx_of = {it: i for i, it in enumerate(D.ITEM)}
    newc = {lab: [] for lab, *_ in M}
    for r in C.itertuples():
        row = src.loc[r.item]
        CON0 = con_of(row)
        i = idx_of.get(r.item)
        k = fold_of.get(r.item, 0)
        if k < 0:
            k = 0
        nobs = np.array([int(row["nobs"])])
        hz = np.array([bool((CON0[0, 24:36] < LZ_BARR_NEUTRAL).any())])
        X = CS.apply_factor(CON0, r.factor, r.level, nobs, hz)
        b = float(vbase_els(X, D.VC[[i]], D.CURVE[[i]], nobs, mode="clark",
                            nstar=np.array([int(r.nstar)]))[0])
        Xi = np.concatenate([X, np.array([[b]], "float32")], 1).astype("float32")
        for lab, pre, res, ft, _ in M:
            Xu = Xi if ft else X
            v = np.mean([don(pre, s, k)(D.CURVE[[i]], D.VC[[i]], Xu)[0] for s in SEEDS])
            newc[lab].append((v + (b if res else 0.0)) * FACE)
    for lab in newc:
        C[lab] = newc[lab]
    C.to_csv(fm.stat("allstar_slope_raw"), index=False, encoding="utf-8-sig")
    out = []
    for ns in (1, 2, 3):
        for f in EN:
            z = C[(C.nstar == ns) & (C.factor == f)]
            if not len(z):
                continue
            z = z.groupby("level")[["mc"] + [l for l, *_ in M]].mean().sort_index()
            zz = z[z.index < 1.0] if f == "barrier" else z
            lo, hi = zz.index.min(), zz.index.max()
            sl = {c: (zz[c][hi] - zz[c][lo]) / (hi - lo) * UNIT[f] for c in z.columns}
            r = {"nstar": ns, "factor": EN[f], "MC": sl["mc"]}
            for lab, *_ in M:
                r[lab] = sl[lab] / sl["mc"]
            out.append(r)
    SL = pd.DataFrame(out).set_index(["nstar", "factor"])
    SL.to_csv(fm.stat("allstar_slope"), encoding="utf-8-sig")
    print()
    print("=== 스타별 기울기 / MC 배수 (1.00 = 정답) ===")
    print(SL.round(2).to_string())
    print()
    for ns in (1, 2, 3):
        s = SL.loc[ns]
        print("  {}스타: ".format(ns) + " | ".join(
            "{} 평균|1-배수| {:.2f} 부호오류 {}".format(lab, np.abs(1 - s[lab]).mean(), int((s[lab] < 0).sum()))
            for lab, *_ in M))

    # ---------- 그림 ----------
    labs = [l for l, *_ in M]; cols_ = [c for *_, c in M]
    fig, ax = plt.subplots(1, 4, figsize=(20.0, 4.6))
    for a, gl in zip(ax[:3], ["1star", "2star", "3star"]):
        v = [T.loc[(gl, l), "MAE"] for l in labs]; e = [T.loc[(gl, l), "MAE_se"] for l in labs]
        b = a.bar(range(len(labs)), v, .6, color=cols_, edgecolor="white", lw=.6,
                  yerr=e, capsize=3, error_kw=dict(lw=1, ecolor="#444444"))
        for r_, val in zip(b, v):
            a.text(r_.get_x() + r_.get_width() / 2, val, "{:.1f}".format(val),
                   ha="center", va="bottom", fontsize=9, fontweight="bold")
        a.set_xticks(range(len(labs))); a.set_xticklabels(SHORT, fontsize=5.4)
        a.set_ylabel("OOS MAE [KRW]")
        a.set_title("{}  (n={:,})".format(gl.replace("star", "-star"),
                                          int(T.loc[(gl, labs[0]), "n"])), fontsize=12)
        a.grid(axis="y", color="#dddddd", lw=.5, alpha=.6); a.margins(y=.2)
    w = .16
    for j, (lab, _, _, _, col) in enumerate(M):
        v = [np.abs(1 - SL.loc[ns][lab]).mean() for ns in (1, 2, 3)]
        ax[3].bar(np.arange(3) + (j - 2) * w, v, w, color=col, edgecolor="white", lw=.6, label=SHORT[j])
        for x, val in zip(np.arange(3) + (j - 2) * w, v):
            ax[3].text(x, val, "{:.2f}".format(val), ha="center", va="bottom", fontsize=7.5)
    ax[3].set_xticks(range(3)); ax[3].set_xticklabels(["1-star", "2-star", "3-star"], fontsize=9)
    ax[3].set_ylabel("mean |1 - slope / MC slope|")
    ax[3].set_title("Contract-term sensitivity", fontsize=12)
    ax[3].legend(fontsize=5.6); ax[3].grid(axis="y", color="#dddddd", lw=.5, alpha=.6); ax[3].margins(y=.25)
    fig.suptitle("Stage-1 extended to 1- and 2-underlying ELS  (same folds, 5 seeds; "
                 "V_base padded to 3 slots by duplication)", fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig(fm.image("allstar")); plt.close(fig)
    headline_fig(LV)
    print()
    print("saved -> allstar.png, headline_3star.png ({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()


def headline_fig(LV):
    """3스타 · 기준 모형 vs 표형 벤치마크 · R²와 MAPE 두 패널.

     README 가 내세우는 그림이다. 수치는 allstar_level.csv 의 group=3star 행 그대로 —
     막대는 단일 모형 5시드 평균, 오차막대는 같은 시드들의 SE 다. 그림 글자는 전부 영어."""
    T = LV[(LV.group == "3star") & (LV.model.isin(HEADLINE))].set_index("model")
    if len(T) != len(HEADLINE):
        print("headline_fig 건너뜀 — 3스타 행이 없다")
        return
    col = {HEADLINE[0]: "#f0c05a", HEADLINE[1]: "#8d99ae"}
    lab = {HEADLINE[0]: "DeepONet\nDirect + V_base + mono",
           HEADLINE[1]: "XGBoost\n(same inputs)"}
    spec = [("R2", "R²  (vs MC price)", "Explained variance", "{:.4f}", 1.0),
            ("MAPE", "OOS MAPE  [%]", "Relative error", "{:.3f}%", None)]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for a, (key, ylab, title, fmt, top) in zip(ax, spec):
        v = [float(T.at[m, key]) for m in HEADLINE]
        e = [float(T.at[m, key + "_se"]) for m in HEADLINE]
        b = a.bar(range(len(HEADLINE)), v, .55, yerr=e, capsize=4,
                  color=[col[m] for m in HEADLINE], edgecolor="white", lw=.8)
        for r, val in zip(b, v):
            a.text(r.get_x() + r.get_width() / 2, val, fmt.format(val),
                   ha="center", va="bottom", fontsize=11, fontweight="bold")
        a.set_xticks(range(len(HEADLINE)))
        a.set_xticklabels([lab[m] for m in HEADLINE], fontsize=9.5)
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=12.5, fontweight="bold")
        a.grid(axis="y", color="#dddddd", lw=.5, alpha=.6)
        a.margins(y=.18)
        if top:
            a.set_ylim(0, top)
    fig.suptitle("3-star ELS — stage-1 accuracy  (n={:,}, mean of 5 single models ± 1 SE)".format(
        int(T.n.iloc[0])), fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(fm.image("headline_3star")); plt.close(fig)
