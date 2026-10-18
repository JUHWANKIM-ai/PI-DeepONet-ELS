# -*- coding: utf-8 -*-
"""tabular 회귀기 (직접 벤치마크 + 하이브리드 stage-2 마진모델 공용). config 파라미터 사용."""
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
import lightgbm as lgb
import xgboost as xgb

from .data import featnum, CAT, time_weights


def load_tab_predictor(D, path):
    """fit_tab 이 저장한 .pkl 을 읽어 예측기로 되살린다 (재학습 없이 forward).

    저장된 **열 목록**으로 D.ml 에서 특성을 다시 꺼낸다 — 그래서 D 의 ml 표가 학습 때와 같은
    열을 가져야 한다. 임의 배열을 넣을 수 있는 쪽은 `load_tab_forward` 다.

    입력
      D    : Dataset — D.ml 에서 저장된 num+cat 열을 꺼낸다
      path : str — .pkl 경로
    출력
      callable(idx) -> ndarray — 행 인덱스 예측기.
                                 ridge 는 저장된 ColumnTransformer 를 먼저 적용한다
    """
    import joblib

    ck = joblib.load(path)
    m = ck["model"]
    ct = ck["ct"]
    X = D.ml[ck["num"] + ck["cat"]]
    if ct is not None:  # ridge: ColumnTransformer 적용
        return lambda idx: m.predict(ct.transform(X.iloc[idx]))
    return lambda idx: m.predict(X.iloc[idx])


def fit_tab(
    D,
    cfg,
    model,
    tr,
    te,
    feat,
    target,
    tw=True,
    va=None,
    save_path=None,
    return_predictor=False,
    half_life=365.25,
    objective=None,
):
    """표형 회귀기 네 종을 같은 규약으로 학습한다 — **학습 폴드에만** fit 하고 시간가중을 건다.

    입력
      D                : Dataset — D.ml 에서 특성을 꺼낸다
      cfg              : dict — cfg["tabular"] 의 모델별 하이퍼파라미터, cfg["train"].es_rounds
      model            : "ridge" | "gbm" | "lgbm" | "xgb"
      tr, te           : ndarray[int] — 학습·예측 행 인덱스
      feat             : "base" | "regime" | "driver" — 수치 특성 묶음 (`data.featnum`)
      target           : ndarray(n,) — **전체 길이** 배열. 이 함수가 tr/va/te 로 자른다
      tw               : bool — 시간가중을 쓸지
      va               : ndarray[int] | None — 검증 행. 비어 있지 않으면 xgb·lgbm 이 조기종료한다
                         (ridge·gbm 은 쓰지 않는다)
      save_path        : str | None — 주면 모델+전처리를 "<save_path>.pkl" 로 저장한다
      return_predictor : bool — True 면 (preds, predictor) 를 돌려준다
      half_life        : float | None — 시간감쇠 반감기 [일]. None 이면 균등
                         (stage-2 ml_resid 가 inner-val 로 골라 넘긴다)
      objective        : str | None — xgb 전용 손실 (예: "reg:pseudohubererror")
    출력
      ndarray                 — return_predictor=False. te 행 예측
      (ndarray, callable)     — return_predictor=True. predictor(idx) 는 D.ml 행 인덱스 예측기
    예외
      ValueError — 모르는 model
    """
    num = featnum(feat)
    X = D.ml[num + CAT]
    Xtr = X.iloc[tr]
    ytr = target[tr]
    w = time_weights(D, tr, half_life) if tw else None
    P = cfg["tabular"]
    es = va is not None and len(va) > 0
    ES_R = cfg["train"].get("es_rounds", 50)
    ct = None
    if es:
        Xva = X.iloc[va]
        yva = target[va]

    if model == "ridge":
        ct = ColumnTransformer(
            [
                (
                    "num",
                    SkPipe(
                        [
                            ("i", SimpleImputer(strategy="median")),
                            ("s", StandardScaler()),
                        ]
                    ),
                    num,
                ),
                ("c", OneHotEncoder(handle_unknown="ignore"), CAT),
            ]
        ).fit(Xtr)
        m = Ridge(alpha=P["ridge"]["alpha"]).fit(ct.transform(Xtr), ytr, sample_weight=w)
        predictor = lambda idx: m.predict(ct.transform(X.iloc[idx]))
    elif model == "gbm":
        m = HistGradientBoostingRegressor(categorical_features=CAT, random_state=0, **P["gbm"])
        m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "lgbm":
        m = lgb.LGBMRegressor(random_state=0, verbose=-1, **P["lgbm"])
        fkw = dict(categorical_feature=CAT, sample_weight=w)
        if es:
            fkw["eval_set"] = [(Xva, yva)]
            fkw["callbacks"] = [lgb.early_stopping(ES_R, verbose=False)]
        m.fit(Xtr, ytr, **fkw)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "xgb":
        xkw = dict(tree_method="hist", enable_categorical=True, random_state=0, **P["xgb"])
        if objective:  # stage-2 강건손실(예: reg:pseudohubererror, 이분산 대응)
            xkw["objective"] = objective
        if es:
            xkw["early_stopping_rounds"] = ES_R
        m = xgb.XGBRegressor(**xkw)
        if es:
            m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
        else:
            m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    else:
        raise ValueError(model)
    if save_path:
        import os, joblib

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(
            {"model_key": model, "model": m, "ct": ct, "num": num, "cat": CAT, "feat": feat},
            save_path + ".pkl",
        )
    preds = predictor(te)
    return (preds, predictor) if return_predictor else preds


def load_tab_forward(path):
    """저장 가중치 -> **임의 입력** 예측기 predict(CURVE, VC, CON) (XGB 앵커 전용).

    앵커 XGB 는 [곡선 | vol·corr | 계약] 을 그대로 이어붙여 학습하므로 (load_tab_predictor 처럼
    D.ml 열을 되읽지 않는다) 배열을 바로 받을 수 있다.

    입력
      path : str — .pkl 경로 (XGB 앵커가 저장한 것)
    출력
      callable(CU, VC, CON) -> ndarray[float32] — 세 블록을 가로로 붙여 예측한다.
        계약을 바꾼 입력을 그대로 넣을 수 있어 민감도 분석에 쓴다
        (`module.anchors.load_xgb` 가 이 함수다)
    """
    import numpy as _np
    import joblib

    m = joblib.load(path)["model"]
    return lambda CU, VC, CON: m.predict(_np.concatenate([CU, VC, CON], axis=1)).astype("float32")
