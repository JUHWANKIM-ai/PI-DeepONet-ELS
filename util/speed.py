# -*- coding: utf-8 -*-
"""예측(추론) 속도 측정 — 학습된 모델로 test 폴드를 예측하는 데 걸리는 시간만 측정.
학습 시간 제외(즉시 견적 지연시간). fold-0 기준, 반복 median.
이론가(MC) 예측 경로 전체를 잰다: 2단계 하이브리드(stage1 앵커 forward + stage2 잔차 predict, 마진 없음)
+ 직접(단일 단계) 벤치마크(ml.csv 전 특성 → MC)."""
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBRegressor

from module.data import time_weights
from module.tabular import fit_tab
from module.train import train_curve
from model.stage2 import xgb_resid, don_resid


def _time(fn: Callable[[], Any], repeat: int = 20, warmup: int = 3) -> float:
    """호출 하나의 소요시간을 중앙값으로 잰다.

    평균이 아니라 중앙값이다 — GC·OS 스케줄링으로 튀는 한두 번이 평균을 끌어올린다.
    warmup 회는 버린다: 지연 초기화와 캐시 워밍이 첫 호출에만 섞인다.

    입력
      fn     : callable — 인수 없이 부를 수 있는 호출. 반환값은 쓰지 않는다
      repeat : int — 계측 횟수
      warmup : int — 버리는 사전 호출 횟수
    출력
      float — 1회 소요시간의 중앙값 [초]
    """
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure(D: SimpleNamespace, cfg: dict[str, Any], repeat: int = 20) -> pd.DataFrame:
    """학습된 모델로 test 폴드를 예측하는 시간만 재서 모델별로 모은다.

    현재 파이프라인 = 이론가(MC) 2단계: y = 앵커(MC_hat) + stage2 잔차(MC−MC_hat), 마진 없음.
    학습 시간은 빠진다 — 재는 것은 **즉시 견적 지연시간**이다. fold-0 기준.

    입력
      D      : Dataset — `module.data` 가 만든 번들. `D.WF[0]` 의 (tr, va, te) 를 쓴다
      cfg    : dict — 설정. tabular.xgb · train.es_rounds · data.time_decay · seed 를 읽는다
      repeat : int — 모델 하나당 계측 횟수 (`_time` 으로 전달)
    출력
      DataFrame — index=model,
                  columns=[kind, n, batch_ms, per_product_us, products_per_sec],
                  per_product_us 오름차순. kind 는 "hybrid"(2단계) | "direct"(단일 단계)
    """
    tr, va, te = D.WF[0]
    N = len(te)
    tw = cfg["data"]["time_decay"]
    rows = []

    def add(
        name: str,
        predict_fn: Callable[[npt.ArrayLike], np.ndarray],
        kind: str = "hybrid",
    ) -> None:
        """예측기 하나를 재서 결과 행을 바깥 rows 에 쌓는다.

        입력
          name       : str — 표에 찍을 모델 이름
          predict_fn : callable(idx) -> ndarray — 행 인덱스를 받아 예측을 내는 호출
          kind       : "hybrid" | "direct" — 2단계인지 단일 단계인지
        출력
          없음 (바깥 rows 에 dict 한 개를 추가한다)
        """
        sec = _time(lambda: predict_fn(te), repeat=repeat)
        rows.append(
            dict(
                model=name,
                kind=kind,
                n=N,
                batch_ms=sec * 1e3,
                per_product_us=sec / N * 1e6,
                products_per_sec=N / sec,
            )
        )

    # stage-1 앵커 (MC 이론가) forward 예측기
    *_, a_mse = train_curve(D, cfg, tr, te, target=D.MC, va=va, return_predict=True)
    *_, a_l1 = train_curve(D, cfg, tr, te, target=D.MC, va=va, loss="l1", return_predict=True)
    *_, a_mape = train_curve(D, cfg, tr, te, target=D.MC, va=va, loss="mape", return_predict=True)
    # XGB 앵커 (matched-입력 = deeponet 블록)
    XP = cfg["tabular"]["xgb"]
    X = D.DON
    w = time_weights(D, tr) if tw else None
    _es = va is not None and len(va) > 0
    _kw = dict(
        n_estimators=XP["n_estimators"],
        learning_rate=XP["learning_rate"],
        max_depth=XP["max_depth"],
        subsample=XP["subsample"],
        colsample_bytree=XP["colsample_bytree"],
        n_jobs=0,
        random_state=cfg["seed"],
    )
    if _es:
        _kw["early_stopping_rounds"] = cfg["train"].get("es_rounds", 50)
    ma = XGBRegressor(**_kw)
    if _es:
        ma.fit(X[tr], D.MC[tr], sample_weight=w, eval_set=[(X[va], D.MC[va])], verbose=False)
    else:
        ma.fit(X[tr], D.MC[tr], sample_weight=w)
    a_xgb = lambda idx: ma.predict(X[idx]).astype("float32")

    # stage-2 잔차모델 (target = 이론가 잔차 MC−MC_hat; mse 앵커 기준으로 학습해 타이밍)
    rt = (D.MC - a_mse(np.arange(D.n))).astype("float32")
    *_, r_xgb = xgb_resid(D, cfg, tr, va, te, rt, return_predict=True)  # 공용 stage2 (4모델)
    *_, r_don = don_resid(D, cfg, tr, va, te, rt, return_predict=True)  # DeepONet stage2 (s2don)

    # 2단계 이론가 예측 = 앵커 + 잔차 (마진 없음)
    add("deeponet_hybrid", lambda idx: a_mse(idx) + r_xgb(idx))
    add("deeponet_hybrid_l1", lambda idx: a_l1(idx) + r_xgb(idx))
    add("deeponet_hybrid_mape", lambda idx: a_mape(idx) + r_xgb(idx))
    add("deeponet_hybrid_s2don", lambda idx: a_mse(idx) + r_don(idx))
    add("xgb_hybrid", lambda idx: a_xgb(idx) + r_xgb(idx))

    # 직접(단일 단계) 벤치마크 = ml.csv 전 특성으로 MC 직접 예측 (앵커/잔차 없음)
    for key, nm in [
        ("ridge", "bench_ridge"),
        ("gbm", "bench_gbm"),
        ("lgbm", "bench_lgbm"),
        ("xgb", "bench_xgboost"),
    ]:
        _, p_dir = fit_tab(
            D, cfg, key, tr, te, "regime", D.MC, tw=bool(tw), va=va, return_predictor=True
        )
        add(nm, p_dir, kind="direct")

    return pd.DataFrame(rows).set_index("model").sort_values("per_product_us")
