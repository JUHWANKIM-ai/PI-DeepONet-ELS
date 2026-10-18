# -*- coding: utf-8 -*-
"""저장된 stage-1 앵커 가중치를 **임의 입력 예측기**로 되살리는 한 곳.

종전에는 `result/models/<name>_fold{k}` 라는 경로 규약이 module/pipeline.py · module/infer.py ·
model/deeponet_ci.py · model/pi_route.py 네 곳에 각각 적혀 있었고, 계약조건을 바꿔 넣는 쓰임은
scratch/contract_sensitivity.py 가 로더를 따로 구현해 쓰고 있었다. 그 둘을 여기로 모은다.

실험 가중치의 이름 규약 (시드 × 폴드):
  <접두사>_deeponet_anchor_seed{s}_fold{k}.pt    DeepONet 앵커
  <접두사>_anchor_seed{s}_fold{k}.pkl            XGB 앵커

`load_curve_predictor`(module/train) 와 다른 점: 이쪽은 D 를 닫아 잡지 않고 (CURVE, VC, CON)
배열을 직접 받는다. 그래서 '계약을 바꾼 입력' 을 표현할 수 있다.
"""
from util import file_manager as fm
from .tabular import load_tab_forward
from .train import load_curve_forward

SEEDS = range(5)
MDIR = fm.RESULT / "models"

load_don = load_curve_forward  # (CURVE, VC, CON) -> 예측
load_xgb = load_tab_forward


def path(prefix, seed, fold, kind="don"):
    """체크포인트 경로를 조립한다 (파일 존재는 확인하지 않는다).

    입력
      prefix : str — 실험 접두사 (예: "ball_dm")
      seed   : int — 학습 시드 0~4
      fold   : int — 폴드 번호
      kind   : 'don'(.pt) | 'xgb'(.pkl) — 중간 토큰과 확장자가 함께 바뀐다
    출력
      str — "result/models/<prefix>_{deeponet_anchor|anchor}_seed{seed}_fold{fold}.{pt|pkl}"
    """
    mid = "deeponet_anchor" if kind == "don" else "anchor"
    ext = ".pt" if kind == "don" else ".pkl"
    return str(MDIR / f"{prefix}_{mid}_seed{seed}_fold{fold}") + ext


def load(prefix, seed, fold, kind="don"):
    """접두사·시드·폴드로 예측기 하나를 만든다.

    `load_curve_predictor`(module/train) 와 달리 D 를 닫아 잡지 않는다 — 돌려받은 예측기에
    (CURVE, VC, CON) 을 직접 넘기므로 '계약을 바꾼 입력' 을 표현할 수 있다.

    입력
      prefix, seed, fold, kind : `path` 와 같다
    출력
      callable(CURVE, VC, CON) -> ndarray — 세 입력 블록을 받아 예측을 내는 forward.
                                            행 인덱스를 받는 예측기가 아니다
    예외
      FileNotFoundError — 해당 체크포인트가 없을 때 (로더가 올린다)
    """
    p = path(prefix, seed, fold, kind)
    return load_don(p) if kind == "don" else load_xgb(p)
