# -*- coding: utf-8 -*-
"""다중 시드 예측 로딩·집계 — 노트북이 **5시드 기준**으로만 지표를 내게 하는 공용 층.

왜 필요한가: DeepONet 앵커의 시드 표준편차가 최종 R² 기준 약 0.0137 인데, 노트북이 비교하던
모델 간 격차(DeepONet 계열 스프레드 0.0222 = 1.6 sd)가 그 안에 들어간다. 단일 시드 숫자로는
순위를 말할 수 없으므로 **모든 지표를 시드 평균 ± sd 로** 낸다.
(근거: scratch/ablation_pi_state.py -> result/statistics/exp_pi_state_ablation.csv)

입력 규약: result/predictions/<model>_seed{s}.csv  (scratch/run_all_seeds.py, run_pi_route_seeds.py 산출)
시드 CSV 가 없는 모델은 조용히 제외한다 — 단일 시드 숫자를 섞지 않는 게 이 모듈의 목적이다.
"""
import numpy as np
import pandas as pd

from util import file_manager as fm
from util.metric import metrics

SEEDS = (0, 1, 2, 3, 4)


def available(models, seeds=SEEDS):
    """시드 CSV 가 **전부** 있는 모델만 반환 (부분 시드는 평균이 편향되므로 제외).

    입력
      models : list[str] — 후보 모델 이름
      seeds  : tuple[int] — 요구 시드. 기본 (0, 1, 2, 3, 4)
    출력
      list[str] — `models` 의 부분집합, 입력 순서 보존
    """
    return [m for m in models if all(fm.prediction(f"{m}_seed{s}").exists() for s in seeds)]


def load(models, seeds=SEEDS):
    """시드별 예측 CSV 를 모두 읽어 사전으로 돌려준다.

    입력
      models : list[str] — 모델 이름
      seeds  : tuple[int] — 시드
    출력
      dict[(model, seed)] -> DataFrame — 파일이 없는 조합은 키 자체가 없다.
                                         ITEM_CD 는 문자열로 고정해 읽는다
    """
    out = {}
    for m in models:
        for s in seeds:
            p = fm.prediction(f"{m}_seed{s}")
            if p.exists():
                out[(m, s)] = pd.read_csv(p, dtype={"ITEM_CD": str})
    return out


def long_metrics(preds, extra=None):
    """시드별 지표를 long 형태로 쌓는다.

    입력
      preds : dict[(model, seed)] -> DataFrame — `load` 의 출력
      extra : callable(DataFrame) -> dict | None — 모델·시드마다 붙일 추가 열
              (예: stage 별 R²)
    출력
      DataFrame — model, seed, n + `util.metric.metrics` 의 열들.
                  mc_pred 열이 있으면 stage1_mc_R2 · stage2_resid_R2 도 붙는다
    """
    rows = []
    for (m, s), d in sorted(preds.items()):
        r = {"model": m, "seed": s, "n": len(d), **metrics(d.y_true, d.y_pred)}
        if "mc_pred" in d.columns:
            r["stage1_mc_R2"] = metrics(d.mc_true, d.mc_pred)["R2"]
            r["stage2_resid_R2"] = metrics(d.resid_true, d.resid_pred)["R2"]
        if extra:
            r.update(extra(d))
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(long_df, keys=("model",), cols=None):
    """long 표를 키별 평균 ± sd 로 접는다.

    입력
      long_df : DataFrame — `long_metrics` 의 출력
      keys    : tuple[str] — 묶을 키. 기본 ("model",)
      cols    : list[str] | None — 집계할 열. None 이면 model/seed/n 을 뺀 수치열 전부
    출력
      DataFrame — index=keys, columns=<열>_mean / <열>_std
    """
    if cols is None:
        cols = [
            c
            for c in long_df.columns
            if c not in ("model", "seed", "n") and pd.api.types.is_numeric_dtype(long_df[c])
        ]
    g = long_df.groupby(list(keys))[list(cols)].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g


def by_group(preds, group_of, min_n=30, metric_cols=("R2", "MAPE%", "RMSE", "Bias%")):
    """그룹(구조·기초자산 등)별 시드 지표 long 표.

    입력
      preds       : dict[(model, seed)] -> DataFrame — `load` 의 출력
      group_of    : callable(DataFrame) -> Series — 행별 그룹 라벨.
                    index 정렬은 호출자 책임이다
      min_n       : int — 이 미만인 그룹은 버린다
      metric_cols : tuple[str] — 표에 담을 지표 이름
    출력
      DataFrame — model, seed, group, n + metric_cols
                  (mc_pred 열이 있으면 stage1_mc_R2 도)
    """
    rows = []
    for (m, s), d in sorted(preds.items()):
        g = group_of(d)
        for key, sub in d.groupby(g.values):
            if len(sub) < min_n:
                continue
            mm = metrics(sub.y_true, sub.y_pred)
            r = {
                "model": m,
                "seed": s,
                "group": key,
                "n": len(sub),
                **{c: mm[c] for c in metric_cols},
            }
            if "mc_pred" in sub.columns:
                r["stage1_mc_R2"] = metrics(sub.mc_true, sub.mc_pred)["R2"]
            rows.append(r)
    return pd.DataFrame(rows)


def paired_delta(long_df, a, b, col="R2"):
    """같은 시드끼리 짝지은 b − a 차이와 요약. 시드 공유가 전제.

    짝짓지 않으면 시드 분산이 모델 간 차이에 그대로 섞인다 — 이 함수가 있는 이유다.

    입력
      long_df : DataFrame — `long_metrics` 의 출력. seed × model 로 pivot 가능해야 한다
      a       : str — 기준 모델
      b       : str — 비교 모델. 차이는 b − a
      col     : str — 비교할 지표 열. 기본 "R2"
    출력
      (Series, dict) — Series: 시드별 차이.
                       dict: mean, sd, se, wins("k/n"), n_seed
    예외
      KeyError — a 나 b 가 표에 없을 때
    """
    p = long_df.pivot(index="seed", columns="model", values=col)
    if a not in p.columns or b not in p.columns:
        raise KeyError(f"{a} / {b} 중 없는 모델")
    d = (p[b] - p[a]).dropna()
    return d, {
        "mean": d.mean(),
        "sd": d.std(),
        "se": d.std() / np.sqrt(len(d)),
        "wins": f"{int((d > 0).sum())}/{len(d)}",
        "n_seed": len(d),
    }


def noise_floor(long_df, col="R2"):
    """arm 내 시드 sd 의 최댓값 — 이보다 작은 모델 간 격차는 해석 불가.

    입력
      long_df : DataFrame — `long_metrics` 의 출력
      col     : str — 지표 열. 기본 "R2"
    출력
      float — 모델별 시드 sd 중 최댓값
    """
    return float(long_df.groupby("model")[col].std().max())
