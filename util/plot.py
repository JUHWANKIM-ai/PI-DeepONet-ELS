# -*- coding: utf-8 -*-
"""그래프 (전부 영문 텍스트). 학술/PPT 용: dpi 400·투명배경·16:9 슬라이드 크기.
각 함수는 파일로 저장하고 fig 를 반환.

*** 기본값(디폴트) ***: 이 모듈을 import 하는 순간 matplotlib 전역 rcParams 가 학술/PPT 사양으로 설정된다.
즉 `from util import plot` 후에는 어떤 `fig.savefig(path)` 든 별도 인자 없이 자동으로
**dpi 400 · 투명배경 · bbox tight** 로 저장되고, 새 figure 기본 크기도 PPT 콘텐츠 밴드(13.333 x 5.0in).
(개별 인자를 주면 그것이 우선.)"""
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, norm

SLIDE = (13.333, 7.5)  # PowerPoint 16:9 widescreen (inches)
SLIDE_W = 13.333  # 슬라이드 폭 (측면 여백 포함 full)
BAND_H = 5.2  # 제목바 + 하단 footnote 를 뺀 콘텐츠 영역 높이 (그림이 밴드에 딱 맞게)
DPI = 400

# ===== 학술 스타일 + PPT 저장 기본값 (import 시 전역 적용) =====
plt.rcParams.update(
    {
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "figure.figsize": (SLIDE_W, 5.0),  # 기본 figure 크기 = PPT 콘텐츠 밴드 (미지정 시)
        # --- 저장 기본값: dpi 400 · 투명 · tight (savefig 인자 없이도 PPT-ready) ---
        "savefig.dpi": DPI,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
    }
)


def _save(fig: plt.Figure, path: str | Path | None, dpi: int = DPI) -> None:
    """투명배경·고해상도 저장 (PPT 오버레이용).

    입력
      fig  : Figure — 저장할 그림
      path : str | Path | None — None 이면 아무것도 하지 않는다 (그림만 쓰고 저장은 안 하는 호출부)
      dpi  : int — 해상도. 기본 DPI(400)
    출력
      없음 (파일만 쓴다)
    """
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)


def model_color(name: str) -> str:
    """모델 이름으로 막대·점 색을 정한다 — 그림 전체에서 계열 색을 일치시키기 위한 규칙.

    이름에 "mape" 가 있으면 진한 파랑, 그 밖에 "deeponet" 이 있으면 연한 파랑,
    나머지는 테라코타(표형 ML 벤치마크)다. 적녹 대비를 쓰지 않는다 (색각 안전).

    입력
      name : str — 모델 이름. 대소문자는 구분하지 않는다
    출력
      str — "#rrggbb" 색 문자열
    """
    n = name.lower()
    if "mape" in n:
        return "#1f6f9c"  # deep blue = DeepONet (MAPE loss)
    if "deeponet" in n:
        return "#5aa9dd"  # blue = DeepONet hybrid
    return "#e07a5f"  # terracotta = tabular ML benchmark (직접/XGB 하이브리드)


def _sort_best_first(df: pd.DataFrame, col: str, mode: str) -> pd.DataFrame:
    """지표 성격에 맞게 '좋은 것부터' 정렬한다.

    입력
      df   : DataFrame — index=model
      col  : str — 정렬 기준 열
      mode : 'higher'(클수록 좋음) | 'lower'(작을수록) | 'abs'(0에 가까울수록)
    출력
      DataFrame — 같은 열 구성, 행 순서만 바뀐 사본
    """
    if mode == "higher":
        return df.sort_values(col, ascending=False)
    if mode == "lower":
        return df.sort_values(col, ascending=True)
    return df.reindex(df[col].abs().sort_values().index)  # abs


def metric_bars(
    metrics_df: pd.DataFrame,
    path: str | Path | None = None,
    title: str = "Walk-forward OOS benchmark",
) -> plt.Figure:
    """모델별 지표를 2×3 가로막대 패널로 그린다.

    패널은 있는 열만 그린다 — 여섯 지표를 다 넘기지 않아도 된다.

    입력
      metrics_df : DataFrame — index=model, columns ⊇ [R2, MAPE%, MAE, RMSE, Bias%, Spearman]
      path       : str | Path | None — 저장 경로. None 이면 저장하지 않는다
      title      : str — 상단 제목
    출력
      Figure — 저장과 별개로 항상 돌려준다 (노트북이 그대로 표시할 수 있게)
    """
    specs = [
        ("R2", "R²  (higher better)", "higher", "{:.3f}"),
        ("MAPE%", "MAPE %  (lower better)", "lower", "{:.2f}"),
        ("Spearman", "Spearman ρ  (higher better)", "higher", "{:.3f}"),
        ("MAE", "MAE  (lower better)", "lower", "{:.4f}"),
        ("RMSE", "RMSE  (lower better)", "lower", "{:.4f}"),
        ("Bias%", "Bias %  (→ 0 better)", "abs", "{:+.2f}"),
    ]
    specs = [s for s in specs if s[0] in metrics_df.columns]
    fig, ax = plt.subplots(2, 3, figsize=(SLIDE_W, BAND_H))
    for a, (col, tt, mode, fmt) in zip(ax.ravel(), specs):
        d = _sort_best_first(metrics_df, col, mode)
        names = list(d.index)
        vals = d[col].values
        y = np.arange(len(names))
        a.barh(y, vals, color=[model_color(x) for x in names], edgecolor="white", linewidth=0.5)
        a.set_yticks(y)
        a.set_yticklabels(names)
        a.invert_yaxis()  # best 맨 위
        a.axvline(0, color="#444", lw=0.6)
        a.set_title(tt)
        a.grid(axis="x", color="#cccccc", lw=0.6, alpha=0.6)
        a.tick_params(labelsize=8.5)
        xr = (vals.max() - min(vals.min(), 0)) or 1
        for i, v in enumerate(vals):
            a.text(
                v + (0.01 * xr if v >= 0 else -0.01 * xr),
                i,
                fmt.format(v),
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=7.5,
            )
        a.margins(x=0.16)
    for a in ax.ravel()[len(specs) :]:
        a.axis("off")
    fig.suptitle(
        title + "   (blue=DeepONet hybrid · orange=ML benchmark)", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)
    return fig


def scatter_grid(
    preds: dict[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    order: Sequence[str] | None = None,
    path: str | Path | None = None,
    ncol: int = 6,
) -> plt.Figure:
    """예측 vs 실제 산점도를 모델별 그리드로 그린다 (콘텐츠 밴드에 맞게 2행 6열).

    입력
      preds : dict[str -> (y_true, y_pred)] — 모델 이름에서 참값·예측 쌍으로
      order : list[str] | None — 그릴 순서. None 이면 preds 의 키 순서
      path  : str | Path | None — 저장 경로
      ncol  : int — 열 수. 행 수는 모델 수로 정해진다
    출력
      Figure — 칸마다 R² 와 Spearman ρ 를 제목에 찍는다
    """
    names = order or list(preds.keys())
    nrow = int(np.ceil(len(names) / ncol))
    h = min(BAND_H, (SLIDE_W / ncol) * nrow * 1.22)
    fig, ax = plt.subplots(nrow, ncol, figsize=(SLIDE_W, h))
    axl = np.atleast_1d(ax).ravel()
    for a, nm in zip(axl, names):
        yt, yp = preds[nm]
        a.scatter(yt, yp, s=4, alpha=0.28, color=model_color(nm), edgecolors="none")
        lim = [float(np.min(yt)), float(np.max(yt))]
        a.plot(lim, lim, "--", color="#c0392b", lw=1.0)
        a.set_title(
            f"{nm}\nR²={r2_score(yt, yp):+.3f} ρ={spearmanr(yt, yp).correlation:+.3f}", fontsize=7.5
        )
        a.set_xlabel("actual", fontsize=8)
        a.set_ylabel("predicted", fontsize=8)
        a.grid(color="#dddddd", lw=0.5, alpha=0.6)
        a.tick_params(labelsize=7)
    for a in axl[len(names) :]:
        a.axis("off")
    fig.suptitle("Predicted vs actual (target) — walk-forward OOS", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)
    return fig


def speed_bar(speed_df: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    """상품 1건당 추론 지연을 로그축 가로막대로 그린다.

    로그축이다 — 2단계 하이브리드와 MC 의 차이가 자릿수 단위라 선형축에서는 한쪽이 안 보인다.

    입력
      speed_df : DataFrame — index=model, per_product_us · products_per_sec 열 필요
                 (`util.speed.measure` 의 출력)
      path     : str | Path | None — 저장 경로
    출력
      Figure — 빠른 것부터 위에서 아래로
    """
    d = _sort_best_first(speed_df, "per_product_us", "lower")
    names = list(d.index)
    vals = d["per_product_us"].values
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(SLIDE_W, min(BAND_H, 0.5 * len(names) + 1.5)))
    ax.barh(y, vals, color=[model_color(x) for x in names], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("inference latency per product (µs, log scale) — lower = faster")
    ax.set_title("Prediction speed — walk-forward fold-0 test (training time excluded)")
    ax.grid(axis="x", which="both", color="#cccccc", lw=0.6, alpha=0.5)
    for i, (v, pps) in enumerate(zip(vals, d["products_per_sec"].values)):
        ax.text(v * 1.08, i, f"{v:.1f} µs  ({pps:,.0f}/s)", va="center", fontsize=8.5)
    ax.margins(x=0.22)
    fig.tight_layout()
    _save(fig, path)
    return fig


# 모델 구분용 정성 팔레트 (grouped bar 에서 시리즈 식별)
_QUAL = ["#1f6f9c", "#5aa9dd", "#2a9d8f", "#e07a5f", "#9b5de5", "#e9c46a", "#c0637a", "#7f8c8d"]


def stage_r2_bar(stage_df: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    """단계별 R² 를 모델 묶음 막대로 그린다 (공정가 2단계).

    키 이름은 하위호환으로 유지: stage1=MC 앵커, stage2=공정가 잔차, final=공정가.

    입력
      stage_df : DataFrame — index=model,
                 columns=[stage1_MC_R2, stage2_resid_R2, final_MC_R2]
                 (`util.metric.stage_r2` 를 모델별로 모은 표)
      path     : str | Path | None — 저장 경로
    출력
      Figure — 범례는 우측 바깥에 둔다 (제목과 겹치지 않게)
    """
    labels = list(stage_df.index)
    x = np.arange(3)
    w = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(SLIDE_W, BAND_H))
    for j, nm in enumerate(labels):
        off = (j - (len(labels) - 1) / 2) * w
        ax.bar(
            x + off,
            stage_df.loc[nm].values,
            w,
            label=nm,
            color=_QUAL[j % len(_QUAL)],
            edgecolor="white",
            linewidth=0.4,
        )
        for i, v in enumerate(stage_df.loc[nm].values):
            ax.text(
                x[i] + off,
                v + (0.012 if v >= 0 else -0.012),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.5,
                rotation=90,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Stage-1\n(anchor: MC theoretical)",
            "Stage-2\n(residual: Fair - MC_hat - margin)",
            "Final\n(fair value)",
        ]
    )
    ax.set_ylabel("R²")
    ax.axhline(0, color="#444", lw=0.6)
    ax.grid(axis="y", color="#cccccc", lw=0.6, alpha=0.5)
    ax.set_title("Per-stage R²  (walk-forward OOS)")
    ax.legend(
        fontsize=9.5, loc="center left", bbox_to_anchor=(1.01, 0.5)
    )  # 우측 범례 = 제목과 겹침 방지
    fig.tight_layout()
    _save(fig, path)
    return fig


# ============================================================================
# 이론가(MC) 예측 세분화 그림 (task#2 발표용). 입력은 이미 병합된 DataFrame m.
#  m 필수 컬럼: y_true, y_pred (+그림별로 y_std/y_lo/y_hi, 'fold', 세분화 축, 'ape', 'cov').
#  magnitude 는 단일 hue 시퀀셜(Blues), 식별은 고정색 — colorblind-safe(적녹 회피).
# ============================================================================
_SEQ = "Blues"  # magnitude 시퀀셜 컬러맵
_MC_BLUE = "#5aa9dd"
_MC_DEEP = "#1f6f9c"
_MC_INK = "#333333"
_MC_GRID = "#cccccc"


def _qbins(x: np.ndarray, k: int) -> tuple[np.ndarray, list[str]]:
    """분위수 경계로 최대 k 구간을 만든다. 중복 경계는 병합한다.

    같은 값이 많은 축(예: 리자드 지급률)에서는 분위수 경계가 겹친다 — 그대로 쓰면 빈 구간이
    생기므로 unique 로 합치고, 그래서 실제 구간 수는 k 보다 적을 수 있다.

    입력
      x : ndarray(n,) — 구간을 나눌 값
      k : int — 요청 구간 수 (상한)
    출력
      (bin_idx, labels) — bin_idx: ndarray(n,) int, 0..K-1
                          labels : list[str] 길이 K, "하단–상단" 형식. K ≤ k
    """
    qs = np.quantile(x, np.linspace(0, 1, k + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    qs = np.unique(qs)
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    labs = [f"{qs[i]:.2f}–{qs[i+1]:.2f}" for i in range(len(qs) - 1)]
    return idx, labs


def mc_error_heatmap(
    m: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    path: str | Path | None = None,
    kx: int = 5,
    ky: int = 5,
    title: str | None = None,
    min_n: int = 15,
) -> plt.Figure:
    """두 연속축(분위수 격자)의 평균 MAPE% 히트맵. 표본 부족 셀은 공백으로 둔다.

    빈 칸은 '오차 0' 이 아니라 '셀 표본이 min_n 미만' 이라는 뜻이다 — 색으로 채우면
    표본 1건짜리 셀이 가장 진하게 보인다.

    입력
      m             : DataFrame — xcol, ycol, 'ape'(절대 백분율 오차) 열 필요
      xcol, ycol    : str — 격자를 만들 두 축의 열 이름
      xlabel, ylabel: str — 축 표시 이름 (영문)
      path          : str | Path | None — 저장 경로
      kx, ky        : int — 축별 요청 구간 수 (`_qbins` 가 줄일 수 있다)
      title         : str | None — None 이면 "MC prediction error by x × y"
      min_n         : int — 이 미만인 셀은 NaN 으로 비운다
    출력
      Figure
    """
    bx, lx = _qbins(m[xcol].values, kx)
    by, ly = _qbins(m[ycol].values, ky)
    kx = len(lx)
    ky = len(ly)
    ape = m["ape"].values
    H = np.full((ky, kx), np.nan)
    for i in range(ky):
        for j in range(kx):
            sel = (by == i) & (bx == j)
            if sel.sum() >= min_n:
                H[i, j] = ape[sel].mean()
    fig, ax = plt.subplots(figsize=(SLIDE_W * 0.62, BAND_H))
    im = ax.imshow(H, origin="lower", cmap=_SEQ, aspect="auto")
    ax.set_xticks(range(kx))
    ax.set_xticklabels(lx, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(ky))
    ax.set_yticklabels(ly, fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    hi = np.nanmax(H)
    for i in range(ky):
        for j in range(kx):
            if not np.isnan(H[i, j]):
                ax.text(
                    j,
                    i,
                    f"{H[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if H[i, j] > hi * 0.6 else _MC_INK,
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("MAPE %  (higher = larger pricing error)", fontsize=9)
    ax.set_title(title or f"MC prediction error by {xlabel} × {ylabel}")
    fig.tight_layout()
    _save(fig, path)
    return fig


def mc_error_by_segment(
    m: pd.DataFrame,
    segs: Sequence[tuple[str, str]],
    path: str | Path | None = None,
    nbin: int = 6,
    title: str = "MC prediction error by contract segment  (walk-forward OOS)",
) -> plt.Figure:
    """세분화 축별 MAPE% 를 2열 막대 그리드로 그린다.

    입력
      m     : DataFrame — 'ape' 열 + segs 가 지목하는 각 열
      segs  : list[(col, label)] — 그릴 축과 표시 이름. 칸 수가 이 길이로 정해진다
      path  : str | Path | None — 저장 경로
      nbin  : int — 축별 분위수 구간 수
      title : str — 상단 제목
    출력
      Figure
    """
    n = len(segs)
    nrow = int(np.ceil(n / 2))
    fig, axs = plt.subplots(nrow, 2, figsize=(SLIDE_W, BAND_H))
    axl = np.atleast_1d(axs).ravel()
    ape = m["ape"].values
    for a, (col, lab) in zip(axl, segs):
        bidx, labs = _qbins(m[col].values, nbin)
        vals = [ape[bidx == i].mean() for i in range(len(labs))]
        x = np.arange(len(labs))
        a.bar(x, vals, color=_MC_BLUE, edgecolor="white", linewidth=0.5)
        a.set_xticks(x)
        a.set_xticklabels(labs, rotation=30, ha="right", fontsize=7.5)
        a.set_ylabel("MAPE %")
        a.set_title(lab, fontsize=11)
        a.grid(axis="y", color=_MC_GRID, lw=0.6, alpha=0.5)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        a.margins(y=0.16)
    for a in axl[n:]:
        a.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
    return fig


def mc_oos_drift(m: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    """walk-forward 폴드(시간)별 R²·MAPE% 라인. 두 축을 겹치지 않고 2패널로 나눈다.

    이중 축은 눈금 선택만으로 추세를 만들어낼 수 있어 쓰지 않는다.

    입력
      m    : DataFrame — 'fold', 'y_true', 'y_pred', 'ape' 열 필요.
             fold 는 0 부터의 정수이고 큰 값이 더 먼 미래다
      path : str | Path | None — 저장 경로
    출력
      Figure — 좌: 폴드별 R², 우: 폴드별 MAPE%
    """
    ks = sorted(m["fold"].unique())
    r2 = [r2_score(m.loc[m.fold == k, "y_true"], m.loc[m.fold == k, "y_pred"]) for k in ks]
    mape = [m.loc[m.fold == k, "ape"].mean() for k in ks]
    xl = [f"Fold {int(k)+1}\n(future slice)" for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(SLIDE_W, BAND_H))
    ax[0].plot(range(len(ks)), r2, marker="o", ms=7, lw=2.0, color=_MC_DEEP)
    ax[0].set_xticks(range(len(ks)))
    ax[0].set_xticklabels(xl, fontsize=9)
    ax[0].set_ylabel("R²")
    ax[0].set_title("R² across time (later fold = further OOS)")
    ax[0].grid(color=_MC_GRID, lw=0.6, alpha=0.5)
    for i, v in enumerate(r2):
        ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax[1].plot(range(len(ks)), mape, marker="s", ms=7, lw=2.0, color=_MC_BLUE)
    ax[1].set_xticks(range(len(ks)))
    ax[1].set_xticklabels(xl, fontsize=9)
    ax[1].set_ylabel("MAPE %")
    ax[1].set_title("MAPE % across time")
    ax[1].grid(color=_MC_GRID, lw=0.6, alpha=0.5)
    for i, v in enumerate(mape):
        ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    fig.suptitle(
        "Out-of-sample stability of MC prediction over the walk-forward horizon",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
    return fig


def mc_ci_calibration(
    m: pd.DataFrame,
    seg_col: str = "sig_eff",
    seg_label: str = "volatility  σ_eff",
    path: str | Path | None = None,
) -> plt.Figure:
    """신뢰구간 캘리브레이션: (좌) reliability curve, (우) 세분화 축별 95% 커버리지.

    대각선 아래면 과신이다. 좌측 곡선은 저장된 σ 로 여섯 신뢰수준을 다시 계산해 그린다.

    입력
      m         : DataFrame — y_true, y_pred, y_std, 'cov' 열 + seg_col
                  ('cov' 는 95% 구간 포함 여부 0/1)
      seg_col   : str — 커버리지를 쪼갤 축의 열 이름. 기본 "sig_eff"
      seg_label : str — 그 축의 표시 이름 (영문)
      path      : str | Path | None — 저장 경로
    출력
      Figure
    """
    err = np.abs(m["y_true"].values - m["y_pred"].values)
    sig = np.maximum(m["y_std"].values, 1e-12)
    nominal = np.array([0.50, 0.68, 0.80, 0.90, 0.95, 0.99])
    z = norm.ppf(0.5 + nominal / 2)
    emp = np.array([(err <= zz * sig).mean() for zz in z])
    fig, ax = plt.subplots(1, 2, figsize=(SLIDE_W, BAND_H))
    ax[0].plot([0, 1], [0, 1], "--", color="#888888", lw=1.4, label="ideal (calibrated)")
    ax[0].plot(nominal, emp, marker="o", ms=7, lw=2.0, color=_MC_DEEP, label="model (aleatoric σ)")
    ax[0].set_xlabel("Nominal coverage")
    ax[0].set_ylabel("Empirical coverage")
    ax[0].set_title("Reliability curve — over-confident if below diagonal")
    ax[0].grid(color=_MC_GRID, lw=0.6, alpha=0.5)
    ax[0].legend(loc="upper left")
    for xN, yE in zip(nominal, emp):
        ax[0].text(
            xN, yE - 0.03, f"{yE*100:.0f}%", ha="center", va="top", fontsize=8, color=_MC_DEEP
        )
    bidx, labs = _qbins(m[seg_col].values, 6)
    cov = [m["cov"].values[bidx == i].mean() * 100 for i in range(len(labs))]
    x = np.arange(len(labs))
    ax[1].bar(x, cov, color=_MC_BLUE, edgecolor="white", linewidth=0.5)
    ax[1].axhline(95, color="#c0392b", lw=1.4, ls="--")
    ax[1].text(
        len(labs) - 0.5, 95, " target 95%", color="#c0392b", va="bottom", ha="right", fontsize=9
    )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labs, rotation=30, ha="right", fontsize=7.5)
    ax[1].set_ylabel("Empirical coverage %")
    ax[1].set_title(f"95% interval coverage by {seg_label}")
    ax[1].grid(axis="y", color=_MC_GRID, lw=0.6, alpha=0.5)
    for i, v in enumerate(cov):
        ax[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Confidence-interval calibration (Gaussian NLL, aleatoric)  — walk-forward OOS",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
    return fig


def mc_pred_ci_band(
    m: pd.DataFrame, path: str | Path | None = None, conf: float = 0.90
) -> plt.Figure:
    """신뢰구간(빨강 밴드)만 표시하고 구간을 벗어난 actual 만 점으로 찍는다 — 과신 가시화.

    구간은 저장된 μ(y_pred)·σ(y_std)로 임의 신뢰수준 conf 에서 재계산한다: μ ± z·σ.
    보정 전 σ 를 그대로 쓰므로 이탈점이 많이 나오는 게 정상이다
    (보정 후 그림은 `mc_pred_ci_calibrated`).

    입력
      m    : DataFrame — y_true, y_pred, y_std 열 필요
      path : str | Path | None — 저장 경로
      conf : float — 신뢰수준 (0~1). 기본 0.90
    출력
      Figure — 제목에 실측 커버리지와 목표를 나란히 찍는다
    """
    z = float(norm.ppf(0.5 + conf / 2))
    d = m.sort_values("y_true").reset_index(drop=True)
    x = np.arange(len(d))
    yt = d["y_true"].to_numpy()
    mu = d["y_pred"].to_numpy()
    sd = d["y_std"].to_numpy()
    lo = mu - z * sd
    hi = mu + z * sd
    mask = (yt < lo) | (yt > hi)  # 구간 밖 actual
    cov = (1 - mask.mean()) * 100
    fig, ax = plt.subplots(figsize=(SLIDE_W, BAND_H))
    ax.fill_between(
        x,
        lo,
        hi,
        color="#e74c3c",
        alpha=0.45,
        linewidth=0,
        label=f"{conf*100:.0f}% interval [μ ± {z:.2f}s]",
    )
    ax.scatter(
        x[mask],
        yt[mask],
        s=14,
        color="#111111",
        alpha=0.8,
        edgecolors="white",
        linewidth=0.2,
        zorder=3,
        label=f"actual outside interval ({mask.mean()*100:.0f}%)",
    )
    ax.set_xlabel("Products sorted by theoretical price (MC)")
    ax.set_ylabel("Price")
    ax.set_title(
        f"MC {conf*100:.0f}% interval & out-of-interval actuals  "
        f"(coverage {cov:.0f}% vs target {conf*100:.0f}%, walk-forward OOS)"
    )
    ax.grid(color=_MC_GRID, lw=0.6, alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)
    return fig


_SPLIT_LABELMAP = {
    "baseline": "baseline",
    "drop_curve": "drop curve",
    "drop_volcorr": "drop vol·corr",
    "drop_contract": "drop contract",
    "baseline_don": "baseline (DeepONet s2)",
    "nllci": "nll-ci (DeepONet s2)",
}


def two_split_bars(
    df: pd.DataFrame,
    path: str | Path | None = None,
    r2_floor: float = -0.5,
    mape_cap: float = 2.0,
    label_map: dict[str, str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """random 7:3 vs walk-forward feature-ablation 비교 (가로 막대 2패널: 좌 R², 우 MAPE%).

    random=미래표본 누수로 낙관적(interpolation), walk-forward=진짜 시간 OOS.
    off-scale(강한 음수 R²·큰 MAPE)는 축을 r2_floor/mape_cap 로 제한하고 막대를 끝까지 그린 뒤
    실제값을 빨강으로 표기한다 — 클립했다는 사실을 숨기지 않는다.

     입력
       df        : DataFrame — ['config','split','R2','MAPE%'],
                   split ∈ {'random7:3','walk-forward'}. config 등장 순서를 그대로 쓴다
       path      : str | Path | None — 저장 경로
       r2_floor  : float — 좌 패널 x축 하한. 이보다 낮은 R² 는 막대를 끝까지 그리고 빨강 표기
       mape_cap  : float — 우 패널 x축 상한. 같은 방식으로 처리
       label_map : dict | None — config 코드 → 표시 이름. 기본 맵에 덮어쓴다
       title     : str | None — None 이면 기본 제목
     출력
       Figure
    """
    lmap = dict(_SPLIT_LABELMAP)
    lmap.update(label_map or {})
    cfgs = list(dict.fromkeys(df["config"]))  # CSV 등장 순서 유지
    y = np.arange(len(cfgs))
    h = 0.38
    BLUE, ORANGE, OFF = _MC_BLUE, "#e07a5f", "#c0392b"

    def val(cfg: str, sp: str, col: str) -> float:
        """(config, split) 한 칸의 값을 꺼낸다.

        입력
          cfg : str — config 이름
          sp  : str — 'random7:3' | 'walk-forward'
          col : str — 'R2' | 'MAPE%'
        출력
          float — 해당 행이 없으면 NaN (호출부가 그 막대를 건너뛴다)
        """
        r = df[(df["config"] == cfg) & (df["split"] == sp)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    fig, ax = plt.subplots(1, 2, figsize=(SLIDE_W, 5.6))
    # 좌: R² (floor 로 클립, off-scale 는 실제값 빨강)
    a = ax[0]
    for cfg, yi in zip(cfgs, y):
        for sp, off, color in [("walk-forward", -h / 2, ORANGE), ("random7:3", h / 2, BLUE)]:
            v = val(cfg, sp, "R2")
            if not np.isfinite(v):
                continue
            drawn = max(v, r2_floor)
            offs = v < r2_floor
            a.barh(yi + off, drawn, height=h, color=color, edgecolor="white", lw=0.5)
            a.text(
                drawn + (-0.012 if drawn < 0 else 0.012),
                yi + off,
                f"{v:.3f}",
                va="center",
                ha="right" if drawn < 0 else "left",
                fontsize=8.5,
                fontweight="bold" if offs else "normal",
                color=OFF if offs else _MC_INK,
            )
    a.axvline(0, color=_MC_INK, lw=0.8)
    a.set_yticks(y)
    a.set_yticklabels([lmap.get(c, c) for c in cfgs], fontsize=9.5)
    a.set_xlim(r2_floor - 0.06, 1.14)
    a.invert_yaxis()
    a.set_xlabel("final MC R²")
    a.set_title("R²  (higher better)")
    a.grid(axis="x", color=_MC_GRID, lw=0.6, alpha=0.5)
    # 우: MAPE% (cap 으로 클립)
    a = ax[1]
    for cfg, yi in zip(cfgs, y):
        for sp, off, color in [("walk-forward", -h / 2, ORANGE), ("random7:3", h / 2, BLUE)]:
            v = val(cfg, sp, "MAPE%")
            if not np.isfinite(v):
                continue
            drawn = min(v, mape_cap)
            offs = v > mape_cap
            a.barh(yi + off, drawn, height=h, color=color, edgecolor="white", lw=0.5)
            a.text(
                drawn + 0.02,
                yi + off,
                f"{v:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold" if offs else "normal",
                color=OFF if offs else _MC_INK,
            )
    a.set_yticks(y)
    a.set_yticklabels([lmap.get(c, c) for c in cfgs], fontsize=9.5)
    a.set_xlim(0, mape_cap + 0.16)
    a.invert_yaxis()
    a.set_xlabel("MAPE %")
    a.set_title("MAPE %  (lower better)")
    a.grid(axis="x", color=_MC_GRID, lw=0.6, alpha=0.5)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=BLUE, label="random 7:3 (interpolation)"),
        Patch(facecolor=ORANGE, label="walk-forward (temporal OOS)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        title
        or "Feature-ablation — R²/MAPE, random 7:3 vs walk-forward  (red = off-scale, true value shown)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, path)
    return fig


def mc_pred_ci_calibrated(
    m: pd.DataFrame, path: str | Path | None = None, conf: float = 0.90
) -> plt.Figure:
    """분산 재보정(variance recalibration) 후 신뢰구간 — 이탈점이 적음을 정직하게 보임.

    raw NLL σ 가 과신이라, held-out 절반에서 보정배수 k=quantile_conf(|actual−μ|/σ) 를 추정하고
    나머지 절반에 μ±k·σ 를 적용해 표시한다. 커버리지도 그 나머지 절반에서 계산하므로
    보정에 쓴 표본으로 자기 성능을 재는 낙관 편향이 없다.

     입력
       m    : DataFrame — y_true, y_pred, y_std 열 필요
       path : str | Path | None — 저장 경로
       conf : float — 목표 신뢰수준 (0~1). 기본 0.90
     출력
       Figure — 홀수 인덱스 절반(표시용)만 그린다. 제목에 σ 배수와 실측 커버리지를 찍는다
    """
    d = m.sort_values("y_true").reset_index(drop=True)
    n = len(d)
    yt = d["y_true"].to_numpy()
    mu = d["y_pred"].to_numpy()
    sd = np.maximum(d["y_std"].to_numpy(), 1e-12)
    zc = np.abs(yt - mu) / sd
    cal = np.arange(n) % 2 == 0
    disp = ~cal  # 짝=보정, 홀=표시/검증
    k = float(np.quantile(zc[cal], conf))  # conf 커버리지를 주는 σ 배수(보정)
    lo = mu - k * sd
    hi = mu + k * sd
    out = (yt < lo) | (yt > hi)
    cov_disp = (1 - out[disp].mean()) * 100
    xd = np.arange(disp.sum())
    dd = d[disp].reset_index(drop=True)
    lo_d, hi_d, yt_d, out_d = lo[disp], hi[disp], yt[disp], out[disp]
    fig, ax = plt.subplots(figsize=(SLIDE_W, BAND_H))
    ax.fill_between(
        xd,
        lo_d,
        hi_d,
        color="#2a9d8f",
        alpha=0.45,
        linewidth=0,
        label=f"calibrated {conf*100:.0f}% interval [μ ± {k:.2f}s]",
    )
    ax.scatter(
        xd[out_d],
        yt_d[out_d],
        s=14,
        color="#111111",
        alpha=0.8,
        edgecolors="white",
        linewidth=0.2,
        zorder=3,
        label=f"actual outside interval ({out_d.mean()*100:.0f}%)",
    )
    ax.set_xlabel("Held-out products sorted by theoretical price (MC)")
    ax.set_ylabel("Price")
    ax.set_title(
        f"MC {conf*100:.0f}% interval after variance recalibration (σ×{k/float(norm.ppf(0.5+conf/2)):.1f})  "
        f"— coverage {cov_disp:.0f}% ≈ target {conf*100:.0f}%, few escapes"
    )
    ax.grid(color=_MC_GRID, lw=0.6, alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)
    return fig
