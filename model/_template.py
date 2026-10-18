# -*- coding: utf-8 -*-
"""새 모델 플러그인 템플릿. 이 파일을 복사해 model/<yourmodel>.py 로 저장하면
registry 가 자동 발견해 2_run 에 편입한다. (파일명이 '_' 로 시작하면 발견에서 제외됨)

계약(contract):
  run(D, cfg) -> dict[str, pandas.DataFrame]
    - key   = 모델 이름(예측 CSV 파일명 stem, result/predictions/<name>.csv).
    - value = walk-forward OOS 예측 DataFrame. 최소 컬럼:
        ITEM_CD, isu_ord, y_true, y_pred   (직접/벤치마크)
      하이브리드는 추가로: mc_true, mc_pred, resid_true, resid_pred
    - 폴드별 가중치 저장은 predict_* 헬퍼가 name= 인자로 처리(result/models/).

D (module.data.load 반환) 주요 필드:
  D.n            상품 수
  D.WF           walk-forward 폴드 [(tr, va, te), ...] (인덱스 배열)
  D.FAIR, D.MC   FAIR 공정가치 / MC 이론가 (float32, 길이 n)
  D.rm           recent_margin (float32, 길이 n)
  D.ITEM, D.ORD  ITEM_CD / isu_ord (길이 n)
  D.VC (n,7)     branch: vol·corr+sig_eff | D.CURVE (n,10) KRW 곡선 | D.CON (n,15) 계약
  D.R, D.TEN, D.SIGEFF  물리용 r / 만기 / sig_eff | D.DEV  torch device
cfg = util.utils.load_config()  (config.yaml)

타깃 규약: 현행 파이프라인은 전부 **공정가(FAIR)** 를 예측한다.
  Fair_hat = MC_hat(stage1 앵커) + recent_margin + Residual_hat(stage2)
  직접 벤치마크 → predict_direct_tab(..., target=D.FAIR).
  하이브리드    → predict_hybrid(..., target=D.FAIR, use_margin=True).
  stage-1 앵커 자체는 여전히 MC 이론가를 근사한다 → train_curve(..., target=D.MC).

재사용 헬퍼:
  module.pipeline.predict_direct_tab(D, cfg, model_key, name=)   # 직접(MC) tabular 벤치마크(단일 단계)
  module.pipeline.predict_hybrid(D, cfg, anchor_fn, resid_fn=None, name=, target=, use_margin=)  # 2단계 하이브리드
  module.train.train_curve              # stage-1 앵커 학습기 (anchor_fn 안에서 사용)
  model.stage2.xgb_resid / don_resid    # stage-2 잔차모델 (predict_hybrid resid_fn=)
"""
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from module.pipeline import predict_hybrid
from module.train import train_curve


def _anchor(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    tr: np.ndarray,
    va: np.ndarray | None,
    te: np.ndarray,
    save_path: str | None = None,
) -> Callable[[npt.ArrayLike], np.ndarray]:
    """stage-1 앵커: MC 이론가를 예측하는 예측기(predictor) 콜백을 반환. (Part B 계약)

    입력
      tr, va, te : ndarray[int] — 이 폴드의 학습·검증·시험 행 인덱스
      save_path  : str | None — 주면 폴드 가중치를 그 경로에 저장한다
      D, cfg     : 공통 번들과 설정
    출력
      callable(idx) -> ndarray[float32] — 행 인덱스를 받아 MC_hat 을 내는 예측기.
                                          학습된 모델 자체가 아니라 **호출 가능한 예측기**다
    """
    *_, predict = train_curve(
        D, cfg, tr, te, target=D.MC, va=va, save_path=save_path, return_predict=True
    )
    return predict


def run(D: SimpleNamespace, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """레지스트리가 부르는 진입점 — 이 파일이 만드는 예측 전부를 돌려준다.

    입력
      D   : Dataset — 주요 필드는 이 파일 모듈 독스트링 참고
      cfg : dict — config.yaml
    출력
      dict[str -> DataFrame] — 키는 예측 CSV 파일명 stem 이자 가중치 이름.
                               저장소 전체에서 **유일**해야 한다 (겹치면 run_all 이 멈춘다)
    """
    # 예: 공정가 2단계 하이브리드 하나. 이름은 유일해야 함(다른 모델과 충돌 금지).
    return {
        "my_model_hybrid": predict_hybrid(
            D, cfg, _anchor, name="my_model_hybrid", target=D.FAIR, use_margin=True
        )
    }
