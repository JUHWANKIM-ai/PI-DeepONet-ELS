# -*- coding: utf-8 -*-
"""PI-DeepONet 하이브리드 — 물리정보 stage-1 앵커 + 기존 stage-2 잔차.

stage-1 앵커만 물리정보(PDE 잔차 + ELS 이벤트 경계)로 바뀌고, stage-2(시장마진 잔차)는
다른 모델과 **동일한** model.stage2.ml_resid 를 쓴다 → PI 효과가 stage-1 에 국소화돼 분리 측정된다.
  Fair_hat = MC_hat(PI 앵커) + recent_margin + Residual_hat(stage-2)

두 개를 함께 학습한다 (순서 고정):
  pi_deeponet_hybrid_data_only  물리항 0 — PI 구조의 대조군
  pi_deeponet_hybrid_pde        + PDE 잔차. 같은 폴드의 data_only 체크포인트에서 커리큘럼 초기화

PDE 를 scratch 에서 학습하면 물리 스케일이 데이터항을 압도해 발산한다. data_only 로 데이터 적합을
먼저 잡고 그 위에 물리를 얹는 순서가 릴리스에서 검증된 레시피이므로 이 순서를 강제한다."""
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from module.pi_train import PREDECESSOR, train_pi_curve
from module.pipeline import predict_hybrid

NAME = "pi_deeponet_hybrid"


def _anchor(variant: str) -> Callable[..., Callable[[npt.ArrayLike], np.ndarray]]:
    """variant 별 stage-1 앵커 콜백을 만든다 (pipeline 규약: anchor_fn(D,cfg,tr,va,te,save_path)).

    입력
      variant : str — "data_only" | "pde". PREDECESSOR 에 선행이 있으면 그 체크포인트에서
                커리큘럼 초기화한다
    출력
      callable — pipeline 이 폴드마다 부르는 앵커 함수
    """

    def anchor(
        D: SimpleNamespace,
        cfg: dict[str, Any],
        tr: np.ndarray,
        va: np.ndarray | None,
        te: np.ndarray,
        save_path: str | None = None,
    ) -> Callable[[npt.ArrayLike], np.ndarray]:
        """이 폴드의 PI 앵커를 학습한다. 선행 variant 가 있으면 그 가중치에서 시작한다.

        선행 체크포인트가 없으면 **조용히 scratch 로 학습하지 않고 멈춘다** — PDE 를 scratch
        에서 돌리면 물리항이 데이터항을 압도해 발산하므로, 없는 것은 오류다.

        입력
          D, cfg     : 공통 번들과 설정. 타깃은 D.MC
          tr, va, te : ndarray[int] — 폴드 인덱스 (te 는 쓰지 않는다. 규약 자리)
          save_path  : str | None — 가중치 저장 접두. 선행 경로를 여기서 유도하므로,
                       None 이면 커리큘럼 초기화도 하지 않는다
        출력
          callable(idx) -> ndarray[float32] — MC_hat 예측기
        예외
          FileNotFoundError — 선행 variant 의 같은 폴드 체크포인트가 없을 때
        """
        init_path = None
        prev = PREDECESSOR.get(variant)
        if prev is not None and save_path is not None:
            # 같은 폴드의 선행 variant 체크포인트 (파일명에서 variant 만 치환)
            cand = Path(str(save_path).replace(f"{NAME}_{variant}_", f"{NAME}_{prev}_") + ".pt")
            if not cand.exists():
                raise FileNotFoundError(
                    f"커리큘럼 선행 체크포인트가 없습니다: {cand}\n"
                    f"  → {NAME}_{prev} 를 먼저 학습해야 합니다 (run() 이 그 순서를 보장합니다)"
                )
            init_path = str(cand)
        predict, _ = train_pi_curve(
            D, cfg, tr, va, variant=variant, save_path=save_path, init_path=init_path
        )
        return predict

    return anchor


def _run_retired(D: SimpleNamespace, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """**은퇴(registry 미발견)**: 이름이 'run' 이 아니므로 model.registry 가 이 모듈을 건너뛴다.

    물리를 전 구조에 일괄 적용한 초기 버전이다. 5시드 ablation 결과 data_only 와 deeponet_hybrid 의
    차이가 노이즈 안(paired t=0.42, p=0.70, 5시드 중 2승)이었고, pde 는 STEP 에서 stage-1 을 떨어뜨렸다.
    구조별로 라우팅하는 model/pi_route.py 가 이를 대체한다. 재현이 필요하면 이 함수를 run 으로 되돌린다.

    입력
      D   : Dataset — 앵커는 D.MC, 최종 타깃은 D.FAIR
      cfg : dict — config.yaml
    출력
      dict[str -> DataFrame] — pi_deeponet_hybrid_data_only, pi_deeponet_hybrid_pde.
                               순서가 중요하다: data_only 4폴드가 모두 저장된 뒤에야
                               pde 가 커리큘럼 초기화를 찾을 수 있다
    """
    T = dict(target=D.FAIR, use_margin=True)  # 공정가 2단계 (다른 하이브리드와 동일)
    out = {}
    # 순서 중요: data_only 4폴드가 모두 저장된 뒤에야 pde 가 커리큘럼 초기화를 찾을 수 있다.
    for variant in ("data_only", "pde"):
        name = f"{NAME}_{variant}"
        out[name] = predict_hybrid(D, cfg, _anchor(variant), name=name, **T)
    return out
