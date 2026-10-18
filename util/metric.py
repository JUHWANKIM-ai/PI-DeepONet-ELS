# -*- coding: utf-8 -*-
"""평가 지표: R2/MAE/RMSE/Spearman + 하이브리드 stage별 R2."""
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


def metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """예측 한 벌의 수준 지표를 한 번에 낸다.

    MAPE 계열은 분모를 |y_true| 로 쓰고 1e-8 로 하한을 둔다 — 가격이 0 에 가까운 상품에서
    분모가 터지는 것을 막는다. Bias% 는 부호를 살려서 과대/과소 방향을 남긴다.

    입력
      y_true : array-like(n,) — 참값. 이 저장소에서는 MC 가격
      y_pred : array-like(n,) — 같은 길이의 예측
    출력
      dict — R2, MAE, RMSE, MAPE%, MdAPE%, Bias%, Spearman
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)  # 절대 백분율 오차
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE%": float(ape.mean() * 100),  # 평균 절대 %오차 (핵심)
        "MdAPE%": float(np.median(ape) * 100),  # 중앙값 %오차 (이상치 강건)
        "Bias%": float(
            ((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)).mean() * 100
        ),  # 부호있는 %편향
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def metrics_by(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, group: npt.ArrayLike, min_n: int = 30
) -> pd.DataFrame:
    """그룹 라벨별 지표 표. 표본 min_n 미만 그룹은 '(small)' 하나로 합친다.

    입력
      y_true : array-like(n,) — 참값
      y_pred : array-like(n,) — 예측
      group  : array-like(n,) — 행별 그룹 라벨. 문자열로 변환해 쓴다
      min_n  : int — 이 미만인 그룹은 '(small)' 로 합친다.
               합친 뒤에도 표본 2 미만인 그룹은 버린다 (R²·Spearman 을 계산할 수 없다)
    출력
      DataFrame — index=group,
                  columns=[n, R2, MAE, RMSE, MAPE%, MdAPE%, Bias%, Spearman], n 내림차순
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    g = pd.Series(np.asarray(group, dtype=object)).astype(str)
    cnt = g.value_counts()
    g = g.where(g.map(cnt) >= min_n, "(small)")
    rows = {}
    for lab in g.unique():
        m = (g == lab).values
        if m.sum() < 2:  # R2/Spearman 계산 불가
            continue
        rows[lab] = {"n": int(m.sum()), **metrics(y_true[m], y_pred[m])}
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int)
    return out.sort_values("n", ascending=False)


def stage_r2(df: pd.DataFrame) -> dict[str, float] | None:
    """2단계 하이브리드 예측에서 단계별 R² 를 낸다.

    직접(단일 단계) 벤치마크엔 stage 열이 없어 None 을 돌려준다 — 호출부는 그 None 으로
    stage_r2 그림을 건너뛴다.

    입력
      df : DataFrame — mc_true/mc_pred/resid_true/resid_pred/y_true/y_pred 열이 모두 필요하다
    출력
      dict | None — stage1_MC_R2, stage2_resid_R2, final_MC_R2.
                    열이 하나라도 없으면 None
    """
    need = {"mc_true", "mc_pred", "resid_true", "resid_pred", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        return None
    return {
        "stage1_MC_R2": r2_score(df["mc_true"], df["mc_pred"]),
        "stage2_resid_R2": r2_score(df["resid_true"], df["resid_pred"]),
        "final_MC_R2": r2_score(df["y_true"], df["y_pred"]),
    }
