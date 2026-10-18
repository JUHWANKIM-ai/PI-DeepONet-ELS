# -*- coding: utf-8 -*-
"""stage-2 가우시안 NLL 신뢰구간 하이브리드 (OptionNet, Lin et al. 2023 의 CI 요소 이식).
기존 deeponet_hybrid_s2don 과 동일한 stage1 MC 앵커 + DeepONet stage2 이되,
stage2 가 평균 μ + 로그분산 logσ² 를 함께 예측(가우시안 NLL)해 [μ−2s, μ+2s] 신뢰구간을 출력.
논문의 MRC(멀티스케일 conv)/LSTM 은 횡단면 ELS 데이터에 시계열 축이 없어 제외; CI 만 이식.

산출: run(D,cfg) -> {"deeponet_hybrid_ci": DataFrame}
  컬럼: ITEM_CD, isu_ord, y_true, y_pred, mc_true, mc_pred, resid_true, resid_pred,
        resid_std, y_std, y_lo, y_hi   (하이브리드 8 + CI 4)
  기존 평가/그림 코드는 여분 컬럼을 무시하므로 안전. CI 리포팅(커버리지 p·밴드)은 이 CSV 로 별도 수행(task#2).
"""
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn

from module.data import to_tensor, zstats, znorm, time_weights
from module.networks import mlp
from module.train import _EarlyStop, _opt, train_curve
from module.pipeline import hybrid_residual_target
from util import file_manager as fm

LAM = 1e-6  # 논문 식17 의 분산 하한 floor: logσ² >= log λ
LV_MAX = 10.0  # 로그분산 상한 (수치 안정)
Z = 2.0  # 신뢰구간 배수 (≈95%, 논문 [μ±2s])


class MarginOperatorNLL(nn.Module):
    """stage-2 잔차 DeepONet(내적) 2-헤드: 평균 μ, 로그분산 logσ².
    branch=vol·corr·sig_eff, trunk=[곡선|계약]. MarginOperator 를 2-헤드로 확장."""

    def __init__(self, nb: int, nt: int, P: int) -> None:
        """두 헤드(μ·logσ²)를 각각 branch·trunk MLP 한 쌍 + 학습 가능한 편향으로 만든다.

        입력
          nb : int — branch 입력 차원 (표준화된 vol·corr·sig_eff 의 열 수)
          nt : int — trunk 입력 차원 ([곡선|계약] 의 열 수)
          P  : int — 내적 잠재 차원 (config networks.P)
        출력
          없음 (파라미터 등록만 한다)
        """
        super().__init__()
        self.b_mu = mlp(nb, P)
        self.t_mu = mlp(nt, P)
        self.b0_mu = nn.Parameter(torch.zeros(1))
        self.b_lv = mlp(nb, P)
        self.t_lv = mlp(nt, P)
        self.b0_lv = nn.Parameter(torch.zeros(1))

    def forward(self, bx: torch.Tensor, tx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """두 헤드를 각각 내적해 평균과 로그분산을 낸다.

        입력
          bx : Tensor(n, nb) — 표준화된 branch 입력
          tx : Tensor(n, nt) — 표준화된 trunk 입력
        출력
          (mu, lv) — 각각 Tensor(n,). lv 는 **clamp 전** 로그분산이다.
                     하한(log LAM)·상한(LV_MAX) 은 손실과 예측 쪽에서 건다
        """
        mu = (self.b_mu(bx) * self.t_mu(tx)).sum(-1) + self.b0_mu
        lv = (self.b_lv(bx) * self.t_lv(tx)).sum(-1) + self.b0_lv
        return mu, lv


def _nll_loss(
    mu: torch.Tensor, lv: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None
) -> torch.Tensor:
    """가우시안 음의 로그우도(표준화 공간). 상수항 0.5·log(2π) 는 최적화에 무관해 생략.

    입력
      mu : Tensor(n,) — 예측 평균
      lv : Tensor(n,) — 예측 로그분산. 여기서 [log LAM, LV_MAX] 로 clamp 한다 (논문 식17 floor)
      y  : Tensor(n,) — 표준화된 참값
      w  : Tensor(n,) | None — 표본가중. None 이면 균등 평균, 주면 가중평균
    출력
      Tensor() — 스칼라 손실
    """
    lvc = lv.clamp(float(np.log(LAM)), LV_MAX)
    per = 0.5 * (lvc + (y - mu) ** 2 * torch.exp(-lvc))
    if w is None:
        return per.mean()
    return (per * w).sum() / w.sum()


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


def _predict_hybrid_ci(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    anchor_fn: Callable[..., Callable[[npt.ArrayLike], np.ndarray]],
    resid_ci_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    name: str | None = None,
    target: np.ndarray | None = None,
    use_margin: bool = True,
) -> pd.DataFrame:
    """CI 2단계 하이브리드 — `pipeline.predict_hybrid` 를 미러링하고 CI 열 4개를 더한다.

    y = MC_hat + recent_margin + mean, 구간 = [y − Z·std, y + Z·std].

    입력
      D, cfg      : 공통 번들과 설정. 폴드는 D.WF 전부를 돈다
      anchor_fn   : callable(D, cfg, tr, va, te, save_path=) -> predict(idx) -> ndarray
                    stage-1 앵커를 만드는 호출
      resid_ci_fn : callable(D, cfg, tr, va, te, target, save_path=) -> (mean_te, std_te)
                    stage-2 를 학습해 평균과 표준편차를 함께 내는 호출
      name        : str | None — 가중치 파일 접두. None 이면 저장하지 않는다
      target      : ndarray | None — 최종 타깃. None 이면 D.FAIR
      use_margin  : bool — False 면 recent_margin 을 0 으로 둔다
    출력
      DataFrame — 폴드를 이어 붙인 OOS 예측. 하이브리드 8열
                  (ITEM_CD, isu_ord, y_true, y_pred, mc_true, mc_pred, resid_true, resid_pred)
                  + CI 4열 (resid_std, y_std, y_lo, y_hi)
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
        mean_te, std_te = resid_ci_fn(
            D, cfg, tr, va, te, rt, save_path=_mp(f"{name}_resid" if name else None, k)
        )
        mc_te = mc_hat[te]
        mean_te = np.asarray(mean_te, dtype="float32")
        y_std = np.asarray(std_te, dtype="float32")
        y_pred = (mc_te + rm_full[te] + mean_te).astype("float32")
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
                    "resid_pred": mean_te,
                    "resid_std": y_std,
                    "y_std": y_std,
                    "y_lo": y_pred - Z * y_std,
                    "y_hi": y_pred + Z * y_std,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def nll_resid(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    tr: np.ndarray,
    va: np.ndarray | None,
    te: np.ndarray,
    target: np.ndarray,
    return_predict: bool = False,
    save_path: str | None = None,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, Callable[[npt.ArrayLike], tuple[np.ndarray, np.ndarray]]]
):
    """stage-2 NLL 잔차: 표준화 공간에서 (μ, logσ²) 를 학습해 원본 스케일 (mean, std) 로 되돌린다.

    `don_resid` 와 같은 블록(branch=VC, trunk=[곡선|계약])·표준화·조기종료 규약을 쓰고
    손실만 NLL 이다. 표준화는 **학습 폴드만**으로 통계를 잡는다 (누출 방지).

    입력
      D, cfg         : 공통 번들과 설정. train.batch · train.nit · networks.P · seed 를 읽는다
      tr, va, te     : ndarray[int] — 폴드 인덱스
      target         : ndarray(n,) — 잔차 타깃. 학습 폴드 평균·표준편차로 표준화한다
      return_predict : bool — True 면 예측기까지 함께 돌려준다
      save_path      : str | None — 주면 "<save_path>.pt" 로 가중치와 표준화 통계를 저장한다
    출력
      (mean_te, std_te)                     — return_predict=False
      (mean_te, std_te, predict)            — return_predict=True
        mean_te, std_te : ndarray[float32] — te 행의 평균·표준편차 (원본 스케일)
        predict         : callable(idx) -> (mean, std)
    """
    dev = D.DEV
    B = cfg["train"]["batch"]
    NIT = cfg["train"]["nit"]
    P = cfg["networks"]["P"]
    torch.manual_seed(cfg["seed"])
    Bmat = D.VC
    Tmat = np.concatenate([D.CURVE, D.CON], axis=1)
    bm, bs = zstats(Bmat, tr)
    tm, ts = zstats(Tmat, tr)
    Bt = to_tensor(znorm(Bmat, bm, bs), dev)
    Tt = to_tensor(znorm(Tmat, tm, ts), dev)
    ym, ysd = float(target[tr].mean()), float(target[tr].std() + 1e-8)
    Y = to_tensor((target - ym) / ysd, dev)
    net = MarginOperatorNLL(Bt.shape[1], Tt.shape[1], P).to(dev)
    opt = _opt(net, cfg)
    trt = torch.tensor(tr, device=dev)
    if cfg["data"]["time_decay"]:
        wsamp = torch.tensor(time_weights(D, tr), device=dev, dtype=torch.float32)
    else:
        wsamp = torch.ones(len(tr), device=dev)

    def _vloss(vi: torch.Tensor) -> torch.Tensor:
        """검증 손실 — 조기종료가 부른다.

        입력
          vi : Tensor[int] — 검증 행 인덱스
        출력
          Tensor() — 가중 없는 NLL (학습 손실과 달리 표본가중을 걸지 않는다)
        """
        mu, lv = net(Bt[vi], Tt[vi])
        return _nll_loss(mu, lv, Y[vi])

    es = _EarlyStop(net, cfg, va, dev)
    for it in range(NIT):
        bb = trt[torch.multinomial(wsamp, B, replacement=True)]
        mu, lv = net(Bt[bb], Tt[bb])
        opt.zero_grad()
        _nll_loss(mu, lv, Y[bb]).backward()
        opt.step()
        if es.step(it, _vloss):
            break
    es.restore()
    net.eval()
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "state": net.state_dict(),
                "P": P,
                "nb": Bt.shape[1],
                "nt": Tt.shape[1],
                "bm": bm,
                "bs": bs,
                "tm": tm,
                "ts": ts,
                "ym": ym,
                "ysd": ysd,
            },
            save_path + ".pt",
        )

    def predict(idx: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """학습된 망으로 평균·표준편차를 원본 스케일로 낸다.

        표준편차에도 ysd 를 곱한다 — 표준화 공간의 σ 를 그대로 쓰면 구간이 원가격 단위에서
        틀린다. 로그분산은 손실과 같은 범위로 clamp 한 뒤 exp 한다.

        입력
          idx : array-like[int] — 예측할 행 인덱스
        출력
          (mean, std) — 각각 ndarray[float32], 원본 스케일
        """
        with torch.no_grad():
            i = torch.tensor(np.asarray(idx), device=dev)
            mu, lv = net(Bt[i], Tt[i])
            lvc = lv.clamp(float(np.log(LAM)), LV_MAX)
            mean = (mu * ysd + ym).cpu().numpy().astype("float32")
            std = (torch.exp(0.5 * lvc) * ysd).cpu().numpy().astype("float32")
            return mean, std

    mean_te, std_te = predict(te)
    return (mean_te, std_te, predict) if return_predict else (mean_te, std_te)


def _anchor(
    D: SimpleNamespace,
    cfg: dict[str, Any],
    tr: np.ndarray,
    va: np.ndarray | None,
    te: np.ndarray,
    save_path: str | None = None,
) -> Callable[[npt.ArrayLike], np.ndarray]:
    """stage-1 앵커: MC 이론가 예측기(train_curve 재사용). deeponet.py 의 `_anchor` 와 동일.

    입력
      D, cfg     : 공통 번들과 설정. 타깃은 D.MC
      tr, va, te : ndarray[int] — 폴드 인덱스
      save_path  : str | None — 주면 폴드 가중치를 저장한다
    출력
      callable(idx) -> ndarray[float32] — MC_hat 예측기
    """
    *_, predict = train_curve(
        D, cfg, tr, te, target=D.MC, va=va, save_path=save_path, return_predict=True
    )
    return predict


def run(D: SimpleNamespace, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """CI 하이브리드 하나를 학습·예측한다.

    입력
      D   : Dataset — 앵커는 D.MC, 최종 타깃은 D.FAIR
      cfg : dict — config.yaml
    출력
      dict[str -> DataFrame] — {"deeponet_hybrid_ci": 하이브리드 8열 + CI 4열}
    """
    # 공정가(FAIR) 2단계 CI 하이브리드 하나: y=MC_hat+recent_margin+μ, 구간=[y−2s, y+2s].
    return {
        "deeponet_hybrid_ci": _predict_hybrid_ci(
            D, cfg, _anchor, nll_resid, name="deeponet_hybrid_ci", target=D.FAIR, use_margin=True
        )
    }
