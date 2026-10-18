# -*- coding: utf-8 -*-
"""walk-forward OOS 예측 조립: 직접(tabular 벤치마크)·하이브리드. 반환 DataFrame.
name 주면 폴드별 학습 모델 가중치를 result/models/<name>_fold{k}.{pt|pkl} 로 저장."""
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .tabular import fit_tab
from model.stage2 import ml_resid
from util import file_manager as fm


def _mp(name: str | None, k: int) -> str | None:
    """폴드별 모델 저장 경로 접두를 만든다. 확장자는 각 학습함수가 붙인다.

    입력
      name : str | None — 모델 이름. None 이면 저장하지 않겠다는 뜻이다
      k    : int — 폴드 번호
    출력
      str | None — "result/models/<name>_fold<k>". name 이 None 이면 None.
                   디렉터리는 이 호출이 만든다
    """
    if name is None:
        return None
    d = fm.RESULT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{name}_fold{k}")


def hybrid_residual_target(
    fair: npt.ArrayLike, mc_hat: npt.ArrayLike, rm: npt.ArrayLike
) -> np.ndarray:
    """stage-2 잔차 타깃 = FAIR − stage1예측(MC_hat) − recent_margin.

    기존엔 MC 참값을 썼으나 stage1 **예측값** 기준으로 재정의했다 → 학습·추론이 일치하고
    stage1 근사오차를 stage2 가 흡수한다.

    입력
      fair   : array-like(n,) — 최종 타깃 (공정가 또는 MC)
      mc_hat : array-like(n,) — stage-1 예측
      rm     : array-like(n,) — recent_margin. 마진을 쓰지 않는 호출은 0 배열을 넘긴다
    출력
      ndarray(n,) float32 — 세 배열의 원소별 차. NaN 은 그대로 전파된다
                            (폴드 밖 행은 호출부가 NaN 으로 두고 쓰지 않는다)
    """
    return (np.asarray(fair) - np.asarray(mc_hat) - np.asarray(rm)).astype("float32")


def predict_direct_tab(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    model_key: str,
    name: str | None = None,
    target: np.ndarray | None = None,
) -> pd.DataFrame:
    """직접(단일 단계) tabular 회귀 벤치마크 — ml.csv 전 특성으로 타깃을 바로 예측(2단계 아님).

    입력
      D, cfg    : 공통 번들과 설정. 폴드는 D.WF 전부를 돈다. 피처셋은 "regime" 고정
      model_key : str — fit_tab 의 모델 키 ("ridge" | "gbm" | "lgbm" | "xgb")
      name      : str | None — 주면 폴드별 .pkl 을 저장한다 (infer 재현용)
      target    : ndarray | None — None 이면 D.MC(이론가)
    출력
      DataFrame — 폴드를 이어 붙인 OOS 예측. ITEM_CD, isu_ord, y_true, y_pred 네 열만
                  (하이브리드의 stage 열은 없다 → `util.metric.stage_r2` 가 None 을 낸다)
    """
    tgt = D.MC if target is None else target
    rows = []
    for k, (tr, va, te) in enumerate(D.WF):
        yp = fit_tab(
            D,
            cfg,
            model_key,
            tr,
            te,
            "regime",
            tgt,
            tw=cfg["data"]["time_decay"],
            va=va,
            save_path=_mp(name, k),
        )
        rows.append(
            pd.DataFrame(
                {"ITEM_CD": D.ITEM[te], "isu_ord": D.ORD[te], "y_true": tgt[te], "y_pred": yp}
            )
        )
    return pd.concat(rows, ignore_index=True)


def predict_hybrid(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    anchor_fn: Callable[..., Callable[[npt.ArrayLike], np.ndarray]],
    resid_fn: Callable[..., tuple[Any, np.ndarray]] | None = None,
    name: str | None = None,
    target: np.ndarray | None = None,
    use_margin: bool = True,
) -> pd.DataFrame:
    """2단계 하이브리드: y = stage1예측(MC_hat) [+ recent_margin] + resid_hat(stage2).

    target=None → D.FAIR (기본, 공정가치 예측). target=D.MC → 이론가 2단계 예측.
    use_margin=True → recent_margin 항 포함(공정가치용). False → 제외(이론가 예측용, 시장마진 없음).
    stage-2 잔차 타깃 = target − MC_hat − (recent_margin if use_margin else 0)  (stage1 예측값 기준, train/val/test 모두).

    입력
      D, cfg     : 공통 번들과 설정. 폴드는 D.WF 전부를 돈다
      anchor_fn  : callable(D, cfg, tr, va, te, save_path=) -> predict(idx) -> ndarray[float32]
                   stage-1 앵커를 만드는 호출
      resid_fn   : callable(D, cfg, tr, va, te, target, save_path=) -> (resid_tr, resid_te) | None
                   None 이면 config 마진모델(`model.stage2.ml_resid`)
      name       : str | None — 주면 stage-1 을 <name>_anchor_fold{k},
                   stage-2 를 <name>_resid_fold{k} 로 저장한다
      target     : ndarray | None — None 이면 D.FAIR(공정가). D.MC 를 주면 이론가 2단계
      use_margin : bool — False 면 recent_margin 을 0 으로 둔다 (이론가 예측용)
    출력
      DataFrame — 폴드를 이어 붙인 OOS 예측 8열:
                  ITEM_CD, isu_ord, y_true, y_pred, mc_true, mc_pred, resid_true, resid_pred
    """
    tgt = D.FAIR if target is None else target
    rm_full = D.rm if use_margin else np.zeros(D.n, dtype="float32")
    rows = []
    for k, (tr, va, te) in enumerate(D.WF):
        anchor_predict = anchor_fn(
            D, cfg, tr, va, te, save_path=_mp(f"{name}_anchor" if name else None, k)
        )
        sel = np.concatenate([tr, va, te]) if len(va) else np.concatenate([tr, te])
        mc_hat = np.full(D.n, np.nan, dtype="float32")
        mc_hat[sel] = np.asarray(anchor_predict(sel), dtype="float32")
        rt = np.full(D.n, np.nan, dtype="float32")
        rt[sel] = hybrid_residual_target(tgt[sel], mc_hat[sel], rm_full[sel])
        # stage-2 잔차모델: 기본=model.stage2.ml_resid(ml 전체특성, cfg['margin']), resid_fn 주면 그것
        rfn = ml_resid if resid_fn is None else resid_fn
        _, resid_pred = rfn(
            D, cfg, tr, va, te, rt, save_path=_mp(f"{name}_resid" if name else None, k)
        )
        mc_te = mc_hat[te]
        y_pred = mc_te + rm_full[te] + resid_pred
        rows.append(
            pd.DataFrame(
                {
                    "ITEM_CD": D.ITEM[te],
                    "isu_ord": D.ORD[te],
                    "y_true": tgt[te],
                    "y_pred": y_pred,
                    "mc_true": D.MC[te],
                    "mc_pred": mc_te,
                    "resid_true": rt[te],
                    "resid_pred": resid_pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
