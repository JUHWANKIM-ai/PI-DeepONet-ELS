# -*- coding: utf-8 -*-
"""머신러닝 벤치마크 — 전부 **이론가(MC) 예측**(fair 아님).
(1) 직접(단일 단계) 벤치마크: ml.csv 전 특성(regime)으로 MC 를 바로 예측 — Ridge/GBM/LightGBM/XGBoost.
(2) XGB 하이브리드(2단계): stage-1 앵커=XGB→MC(MC_hat), stage-2=XGB→이론가 잔차(MC_true−MC_hat), 마진 없음.
    앵커 입력을 DeepONet 앵커와 동일하게 맞춤(순수 아키텍처 비교): 곡선 u0-9 + vol·corr + 전체 계약(strk0-11,B,coupon,tenor).
→ 직접 vs 2단계 하이브리드 비교."""
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBRegressor

from module.pipeline import predict_hybrid, predict_direct_tab
from module.data import time_weights

# 직접(단일 단계) 벤치마크: fit_tab model_key -> 예측 CSV/가중치 이름
_DIRECT = {"ridge": "bench_ridge", "gbm": "bench_gbm", "lgbm": "bench_lgbm", "xgb": "bench_xgboost"}


def _xgb_anchor(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    tr: np.ndarray,
    va: np.ndarray | None,
    te: np.ndarray,
    save_path: str | None = None,
) -> Callable[[npt.ArrayLike], np.ndarray]:
    """stage-1 앵커: XGBoost가 무차익가 MC를 예측.

    입력을 DeepONet-Curve 앵커와 동일하게 맞춤 → 순수 아키텍처 비교.
    X = [곡선 u0-9 | vol·corr(7) | 계약(strk0-11 + B + coupon + tenor)].

    입력
      D          : Dataset — D.CURVE · D.VC · D.CON 을 가로로 붙여 X 로 쓴다. 타깃은 D.MC
      cfg        : dict — tabular.xgb 하이퍼파라미터, data.time_decay, train.es_rounds, seed
      tr, va, te : ndarray[int] — 폴드 인덱스. va 가 비어 있지 않으면 조기종료를 건다
                   (te 는 이 함수가 쓰지 않는다 — 호출 규약을 맞추기 위한 자리다)
      save_path  : str | None — 주면 "<save_path>.pkl" 로 모델을 저장한다
    출력
      callable(idx) -> ndarray[float32] — 행 인덱스를 받아 MC_hat 을 내는 예측기
    """
    XP = cfg["tabular"]["xgb"]
    X = np.concatenate([D.CURVE, D.VC, D.CON], axis=1)
    w = time_weights(D, tr) if cfg["data"]["time_decay"] else None
    es = va is not None and len(va) > 0
    kw = dict(
        n_estimators=XP["n_estimators"],
        learning_rate=XP["learning_rate"],
        max_depth=XP["max_depth"],
        subsample=XP["subsample"],
        colsample_bytree=XP["colsample_bytree"],
        n_jobs=0,
        random_state=cfg["seed"],
    )
    if es:
        kw["early_stopping_rounds"] = cfg["train"].get("es_rounds", 50)
    m = XGBRegressor(**kw)
    if es:
        m.fit(X[tr], D.MC[tr], sample_weight=w, eval_set=[(X[va], D.MC[va])], verbose=False)
    else:
        m.fit(X[tr], D.MC[tr], sample_weight=w)
    if save_path:
        import os, joblib

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"model": m, "kind": "xgb_anchor_curve_vc_con"}, save_path + ".pkl")
    return lambda idx: m.predict(X[idx]).astype("float32")


def run(D: SimpleNamespace, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """표형 ML 벤치마크 전부를 학습·예측한다 — 직접 4종 + XGB 하이브리드 1종.

    입력
      D   : Dataset — 직접 벤치마크는 ml.csv 전 특성(regime), 하이브리드는 CURVE·VC·CON
      cfg : dict — config.yaml
    출력
      dict[str -> DataFrame] — bench_ridge, bench_gbm, bench_lgbm, bench_xgboost,
                               xgb_hybrid. 5개 모두 공정가(FAIR)를 타깃으로 한다
    """
    out = {}
    # (1) 직접(단일 단계) 벤치마크: ml.csv 전 특성으로 공정가(FAIR) 직접 예측
    for key, name in _DIRECT.items():
        out[name] = predict_direct_tab(D, cfg, key, name=name, target=D.FAIR)
    # (2) XGB 하이브리드(2단계): XGB 앵커(MC_hat) + recent_margin + XGB 잔차(FAIR−MC_hat−rm)
    out["xgb_hybrid"] = predict_hybrid(
        D, cfg, _xgb_anchor, name="xgb_hybrid", target=D.FAIR, use_margin=True
    )
    return out
