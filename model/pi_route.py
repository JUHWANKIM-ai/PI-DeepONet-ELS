# -*- coding: utf-8 -*-
"""구조별 라우팅 하이브리드 — LIZARD no-KI 만 PI(PDE) 앵커, 나머지는 DeepONet 앵커.

근거(5_segment 5-1b 실측): PDE 잔차를 얹으면 stage-1 이 STEP 에서는 나빠지고(−0.006~−0.009)
LIZARD 에서만 좋아졌다(no-KI +0.044). STEP 은 표본 8.7k~12.3k 에 stage-1 R² 가 이미 0.96~0.97 로
포화라 물리 prior 가 편향만 더하고, LIZARD no-KI 는 표본 1.9k 에 stage-1 이 0.84 에 그쳐 여지가 있다.
그래서 물리를 **전 구조에 일괄 적용하지 않고** 그 구조에만 라우팅한다.

두 앵커 (둘 다 fold train 전체로 학습 — 라우팅은 예측 선택만 한다):
  branch A  PIStateDeepONet(trunk 61) + state 마스킹 0 + 물리 없음
            = 기능적으로 DeepONet. trunk fan_in 61 이라 초기화가 작아져 **시드 분산이 준다**
              (실측 최종 R² 시드 sd 0.0118 -> 0.0031; scratch/ablation_pi_state.py A1 vs A2)
  branch B  PIStateDeepONet(trunk 61) + state 활성 + PDE 잔차, branch A 체크포인트에서 커리큘럼 초기화
            → LIZARD no-KI 상품에만 사용

조립: mc_hat = where(LIZARD no-KI, branchB, branchA) → stage-2(ml_resid) 하나 → y = mc_hat + rm + resid.
라우팅 키는 module.pi_train.lizard_no_ki_mask (계약 조건만, 누출 없음).
"""
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from module.pi_train import lizard_no_ki_mask, load_pi_predictor, train_pi_curve
from module.pipeline import hybrid_residual_target
from model.stage2 import ml_resid
from util import file_manager as fm

NAME = "deeponet_hybrid_pi_lizard"


def _paths(name: str, seed: int, k: int) -> tuple[str, str, str]:
    """이 (이름, 시드, 폴드) 조합의 가중치 경로 셋을 만든다.

    입력
      name : str — 모델 이름
      seed : int — 학습 시드. 경로에 들어가므로 시드별 가중치가 섞이지 않는다
      k    : int — 폴드 번호
    출력
      (base, pi, resid) — 각각 str. branch A 앵커 · branch B(PI) 앵커 · stage-2 경로 접두.
                          확장자는 각 학습함수가 붙인다. 디렉터리는 이 호출이 만든다
    """
    d = fm.RESULT / "models"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_seed{seed}"
    return (
        str(d / f"{stem}_anchor_base_fold{k}"),
        str(d / f"{stem}_anchor_pi_fold{k}"),
        str(d / f"{stem}_resid_fold{k}"),
    )


def _assemble(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    mc_hat: np.ndarray,
    route: np.ndarray,
    save_path: str | None,
) -> pd.DataFrame:
    """라우팅된 mc_hat 위에 stage-2 를 올려 이 폴드의 OOS 행을 조립한다.

    입력
      D, cfg     : 공통 번들과 설정
      tr, va, te : ndarray[int] — 폴드 인덱스
      mc_hat     : ndarray(n,) float32 — 라우팅이 끝난 stage-1 예측.
                   tr·va·te 밖 행은 NaN 이어도 된다 (쓰지 않는다)
      route      : ndarray(n,) bool — True 면 PI 앵커를 쓴 행. 결과의 route 열이 된다
      save_path  : str | None — stage-2 가중치 경로. None 이면 저장하지 않는다
    출력
      DataFrame — te 행만. pipeline.predict_hybrid 와 같은 8열 + route 열
                  ("pi_pde" | "deeponet")
    """
    sel = np.concatenate([tr, va, te])
    rt = np.full(D.n, np.nan, "float32")
    rt[sel] = hybrid_residual_target(D.FAIR[sel], mc_hat[sel], D.rm[sel])
    _, resid = ml_resid(D, cfg, tr, va, te, rt, save_path=save_path)
    return pd.DataFrame(
        {
            "ITEM_CD": D.ITEM[te],
            "isu_ord": D.ORD[te],
            "y_true": D.FAIR[te],
            "y_pred": mc_hat[te] + D.rm[te] + resid,
            "mc_true": D.MC[te],
            "mc_pred": mc_hat[te],
            "resid_true": rt[te],
            "resid_pred": resid,
            "route": np.where(route[te], "pi_pde", "deeponet"),
        }
    )


def predict_routed(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    name: str = NAME,
    seed: int | None = None,
    with_control: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """4폴드 walk-forward OOS 예측 (pipeline.predict_hybrid 스키마 + route 열).

    with_control=True 면 (routed, control) 을 함께 반환한다. control 은 **같은 branch A 가중치**로
    라우팅만 끈 것 — 앵커를 공유하므로 시드·초기화·데이터가 완전히 동일한 짝지은 대조군이고,
    추가 비용은 폴드당 stage-2 한 번뿐이다. (= ablation 의 A2_width 와 같은 설정)

    입력
      D, cfg       : 공통 번들과 설정
      name         : str — 가중치·CSV 이름. 기본 NAME
      seed         : int | None — 학습 시드. None 이면 cfg["seed"]
      with_control : bool — True 면 짝지은 대조군을 함께 돌려준다
    출력
      DataFrame                  — with_control=False. 4폴드를 이어 붙인 OOS 예측
      (routed, control)          — with_control=True. 두 표의 스키마는 같고,
                                   control 은 route 열이 있어도 전부 branch A 예측이다
    """
    seed = cfg["seed"] if seed is None else seed
    route = lizard_no_ki_mask(D)
    rows, crows = [], []
    for k, (tr, va, te) in enumerate(D.WF):
        pa, pb, pr = _paths(name, seed, k)
        # branch A: DeepONet 상당 (state 마스킹) — 분산 축소 초기화
        base, _ = train_pi_curve(D, cfg, tr, va, variant="data_only", save_path=pa, mask_state=True)
        # branch B: PI + PDE, branch A 에서 커리큘럼 초기화
        pi, _ = train_pi_curve(D, cfg, tr, va, variant="pde", save_path=pb, init_path=pa + ".pt")

        sel = np.concatenate([tr, va, te])
        mc_base = np.full(D.n, np.nan, "float32")
        mc_base[sel] = np.asarray(base(sel), dtype="float32")
        mc_hat = mc_base.copy()
        lz = sel[route[sel]]  # LIZARD no-KI 만 PI 앵커로 덮어쓴다
        if len(lz):
            mc_hat[lz] = np.asarray(pi(lz), dtype="float32")

        rows.append(_assemble(D, cfg, tr, va, te, mc_hat, route, pr))
        if with_control:
            crows.append(_assemble(D, cfg, tr, va, te, mc_base, route, None))
    routed = pd.concat(rows, ignore_index=True)
    if with_control:
        return routed, pd.concat(crows, ignore_index=True)
    return routed


def load_routed_predictor(
    D: SimpleNamespace, name: str, seed: int, k: int
) -> Callable[[npt.ArrayLike], np.ndarray]:
    """저장 가중치로 라우팅된 stage-1 예측기를 복원한다 (infer 재현 경로).

    학습을 다시 하지 않는다 — 두 앵커 가중치를 읽어 라우팅만 재현한다.

    입력
      D    : Dataset — 라우팅 키(`lizard_no_ki_mask`)와 입력 행렬을 여기서 얻는다
      name : str — 가중치 이름
      seed : int — 학습 시드
      k    : int — 폴드 번호
    출력
      callable(idx) -> ndarray[float32] — 라우팅까지 반영한 MC_hat 예측기
    """
    pa, pb, _ = _paths(name, seed, k)
    base = load_pi_predictor(D, pa + ".pt")
    pi = load_pi_predictor(D, pb + ".pt")
    route = lizard_no_ki_mask(D)

    def predict(idx: npt.ArrayLike) -> np.ndarray:
        """branch A 로 전부 예측한 뒤 라우팅 대상 행만 branch B 로 덮어쓴다.

        입력
          idx : array-like[int] — 예측할 행 인덱스
        출력
          ndarray[float32] — 길이 len(idx)
        """
        idx = np.asarray(idx)
        out = np.asarray(base(idx), dtype="float32").copy()
        m = route[idx]
        if m.any():
            out[m] = np.asarray(pi(idx[m]), dtype="float32")
        return out

    return predict


def run(D: SimpleNamespace, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """라우팅 하이브리드 하나를 학습·예측한다.

    입력
      D   : Dataset — 앵커는 D.MC, 최종 타깃은 D.FAIR
      cfg : dict — config.yaml
    출력
      dict[str -> DataFrame] — {NAME: 하이브리드 8열 + route 열}
    """
    return {NAME: predict_routed(D, cfg)}
