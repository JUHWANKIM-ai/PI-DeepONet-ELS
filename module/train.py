# -*- coding: utf-8 -*-
"""연산자망 학습 (stage-1 앵커).
train_curve : DeepONet-Curve (CNN 곡선+vol·corr, trunk=계약). target=MC(이론가 앵커) 또는 FAIR.
_EarlyStop/_opt 은 stage-2 (model.stage2) 에서도 공용으로 import 한다.
"""
import numpy as np
import torch

from .data import to_tensor, zstats, znorm, ZCLIP
from .networks import CurveOperatorV2, CurveOperatorMoE


def _opt(net, cfg):
    """공용 옵티마이저 — 모든 망이 같은 설정을 쓰도록 한 곳에 둔다.

    입력
      net : nn.Module — 학습할 망
      cfg : dict — cfg["train"] 의 lr · weight_decay
    출력
      torch.optim.Adam
    """
    return torch.optim.Adam(
        net.parameters(), cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )


class _EarlyStop:
    """val 손실 기반 조기종료 + 최적 가중치 복원. va 비면 비활성."""

    def __init__(self, net, cfg, va, dev):
        """검증 행이 있으면 활성, 없으면 **아무 일도 하지 않는 상태**로 둔다.

        입력
          net : nn.Module — 가중치를 저장·복원할 대상
          cfg : dict — cfg["train"] 의 es_every(기본 100) · es_patience(기본 6)
          va  : array-like[int] | None — 검증 행. 비면 비활성
          dev : str — torch 장치
        출력
          없음
        """
        self.on = va is not None and len(va) > 0
        self.net = net
        self.every = cfg["train"].get("es_every", 100)
        self.patience = cfg["train"].get("es_patience", 6)
        self.vat = torch.tensor(np.asarray(va), device=dev) if self.on else None
        self.best = float("inf")
        self.best_state = None
        self.wait = 0

    def step(self, it, val_loss_fn):
        """매 스텝 호출한다. es_every 의 배수일 때만 실제로 검증한다.

        입력
          it           : int — 0-기반 스텝 번호. (it+1) 이 es_every 의 배수일 때만 잰다
          val_loss_fn  : callable(vat) -> 스칼라 — 검증 손실. no_grad 안에서 부른다
        출력
          bool — True 면 **중단**(patience 소진). 비활성이거나 평가 스텝이 아니면 False.
                 개선된 스텝에서는 가중치 사본을 보관한다
        """
        if not self.on or (it + 1) % self.every != 0:
            return False
        with torch.no_grad():
            vloss = float(val_loss_fn(self.vat))
        if vloss < self.best - 1e-7:
            self.best = vloss
            self.best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience

    def restore(self):
        """보관해 둔 최적 가중치를 망에 되돌린다.

        학습 루프가 끝나면 **항상** 부른다. 비활성이거나 보관본이 없으면 아무 일도 하지 않으므로
        호출부가 조건을 따질 필요가 없다.

        입력
          없음
        출력
          없음 (망의 상태를 바꾼다)
        """
        if self.on and self.best_state is not None:
            self.net.load_state_dict(self.best_state)


# ===== DeepONet-Curve (데이터기반, stage-1 앵커) =====
def _loss_fn(loss):
    """정규화 스케일 손실을 고른다.

    입력
      loss : "mse" | "l1" | "huber" — 그 밖의 값은 **조용히 mse** 가 된다.
             "mape" 는 원본 스케일이라 여기서 다루지 않고 `train_curve` 가 따로 처리한다
    출력
      callable(pred, y) -> 스칼라
    """
    if loss == "l1":
        return lambda p, y: (p - y).abs().mean()
    if loss == "huber":
        return lambda p, y: torch.nn.functional.huber_loss(p, y, delta=1.0)
    return lambda p, y: ((p - y) ** 2).mean()


def train_curve(
    D,
    cfg,
    tr,
    te,
    target,
    va=None,
    loss="mse",
    return_predict=False,
    save_path=None,
    moe_k=0,
    moe_ib=0,
):
    """DeepONet-Curve 를 학습한다 — stage-1 앵커의 정본 학습기.

    표준화 통계(입력 세 블록 + 타깃)는 **학습 폴드만**으로 잡는다. 타깃도 표준화해 학습하고
    예측에서 원본 스케일로 되돌린다.

    입력
      D, cfg         : 공통 번들과 설정. train.batch · train.nit · networks.P · seed
      tr, te         : ndarray[int] — 학습·예측 행
      target         : ndarray(n,) — D.MC(이론가 앵커) 또는 D.FAIR(직접)
      va             : ndarray[int] | None — 검증 행. 주면 조기종료가 켜진다
      loss           : "mse" | "l1" | "huber" | "mape".
                       mape 만 원본 스케일 상대오차로 계산한다
      return_predict : bool — True 면 예측기까지 돌려준다
      save_path      : str | None — "<...>.pt" 로 가중치와 표준화 통계를 저장한다
                       (이미 .pt 로 끝나면 그대로)
      moe_k          : int — 0 이면 CurveOperatorV2, >0 이면 CurveOperatorMoE(K=moe_k)
      moe_ib         : int — MoE 배리어 게이트가 볼 계약 열 인덱스
    출력
      (pred_tr, pred_te)          — return_predict=False
      (pred_tr, pred_te, predict) — return_predict=True.
                                    predict(idx) 는 원본 스케일 예측기
    """
    dev = D.DEV
    B = cfg["train"]["batch"]
    NIT = cfg["train"]["nit"]
    P = cfg["networks"]["P"]
    torch.manual_seed(cfg["seed"])
    um, us = zstats(D.CURVE, tr)
    vm, vs = zstats(D.VC, tr)
    cm, cs = zstats(D.CON, tr)
    Un = to_tensor(znorm(D.CURVE, um, us), dev)
    Vn = to_tensor(znorm(D.VC, vm, vs), dev)
    Cn = to_tensor(znorm(D.CON, cm, cs), dev)
    net = (
        CurveOperatorMoE(Vn.shape[1], Cn.shape[1], P, K=moe_k, ib=moe_ib)
        if moe_k
        else CurveOperatorV2(Vn.shape[1], Cn.shape[1], P)
    ).to(dev)
    opt = _opt(net, cfg)
    ym, ysd = float(target[tr].mean()), float(target[tr].std() + 1e-8)
    Y = to_tensor((target - ym) / ysd, dev)
    Torig = to_tensor(target, dev)  # 원본 스케일 타깃 (mape loss용)
    ntr = len(tr)
    trt = torch.tensor(tr, device=dev)
    lf = _loss_fn(loss)

    def bloss(pr, idx):
        """배치 손실. mape 만 역정규화해 원본 스케일 상대오차로 잰다.

        입력
          pr  : Tensor(b,) — 표준화 공간의 예측
          idx : Tensor[int] — 그 배치의 행 인덱스 (타깃을 여기서 집는다)
        출력
          Tensor() — 스칼라 손실
        """
        if loss == "mape":
            po = pr * ysd + ym
            return ((po - Torig[idx]).abs() / Torig[idx].abs().clamp(min=1e-6)).mean()
        return lf(pr, Y[idx])

    es = _EarlyStop(net, cfg, va, dev)
    for it in range(NIT):
        bb = trt[torch.randint(0, ntr, (B,), device=dev)]
        pr = net.V(Un[bb], Vn[bb], Cn[bb])
        opt.zero_grad()
        bloss(pr, bb).backward()
        opt.step()
        if es.step(it, lambda vi: bloss(net.V(Un[vi], Vn[vi], Cn[vi]), vi)):
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
                "nvc": Vn.shape[1],
                "ncon": Cn.shape[1],
                "um": um,
                "us": us,
                "vm": vm,
                "vs": vs,
                "cm": cm,
                "cs": cs,
                "ym": ym,
                "ysd": ysd,
            },
            save_path if save_path.endswith(".pt") else save_path + ".pt",
        )

    def p(idx):
        """학습된 망으로 원본 스케일 예측을 낸다.

        입력
          idx : array-like[int] — 행 인덱스
        출력
          ndarray — 길이 len(idx). 표준화를 되돌린 값(× ysd + ym)
        """
        with torch.no_grad():
            i = torch.tensor(idx, device=dev)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()

    return (p(tr), p(te), p) if return_predict else (p(tr), p(te))


def load_curve_predictor(D, path):
    """저장된 DeepONet-Curve 가중치를 **행 인덱스 예측기**로 되살린다 (forward 만, 재학습 없음).

    체크포인트에 든 표준화 통계를 그대로 쓴다 — 지금 데이터로 다시 계산하면 학습 때와
    다른 공간에서 예측하게 된다. D 를 닫아 잡으므로 '계약을 바꾼 입력' 은 표현할 수 없다
    (그 쓰임은 `load_curve_forward`).

    입력
      D    : Dataset — CURVE · VC · CON 을 표준화해 미리 올려 둔다
      path : str — .pt 경로
    출력
      callable(idx) -> ndarray — 원본 스케일 예측기
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CurveOperatorV2(ck["nvc"], ck["ncon"], ck["P"]).to(D.DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    Un = to_tensor(znorm(D.CURVE, ck["um"], ck["us"]), D.DEV)
    Vn = to_tensor(znorm(D.VC, ck["vm"], ck["vs"]), D.DEV)
    Cn = to_tensor(znorm(D.CON, ck["cm"], ck["cs"]), D.DEV)
    ym, ysd = ck["ym"], ck["ysd"]

    def predict(idx):
        """복원된 망으로 원본 스케일 예측을 낸다.

        입력
          idx : array-like[int] — 행 인덱스
        출력
          ndarray — 길이 len(idx)
        """
        with torch.no_grad():
            i = torch.tensor(np.asarray(idx), device=D.DEV)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()

    return predict


def load_curve_forward(path, dev=None, chunk=16384):
    """저장 가중치 -> **임의 입력** 예측기 predict(CURVE, VC, CON) -> np.ndarray.

    load_curve_predictor 는 D 를 닫아 잡고 행 인덱스만 받으므로 '계약을 바꾼 입력' 을 표현할 수 없다.
    계약조건 민감도처럼 CON 을 직접 만들어 넣어야 하는 쓰임은 이쪽을 쓴다.
    정규화 통계는 체크포인트에 든 **학습 폴드 기준** 값을 그대로 적용한다.

    입력
      path  : str — .pt 경로
      dev   : str | None — torch 장치. None 이면 CUDA 가용 시 CUDA
      chunk : int — 한 번에 forward 할 행 수 (메모리 상한)
    출력
      callable(CURVE, VC, CON) -> ndarray — 세 블록을 **원 스케일로** 받아
                                            원본 스케일 예측을 낸다
    """
    import numpy as _np

    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    net = CurveOperatorV2(ck["nvc"], ck["ncon"], ck["P"]).to(d)
    net.load_state_dict(ck["state"])
    net.eval()

    def predict(CU, VC, CON):
        """임의 입력 세 블록을 받아 chunk 단위로 forward 한다.

        입력
          CU  : ndarray(n, 10) — 원 스케일 곡선
          VC  : ndarray(n, nvc) — 원 스케일 vol·corr
          CON : ndarray(n, ncon) — 원 스케일 계약
        출력
          ndarray(n,) — 원본 스케일 예측
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(CU), chunk):
                U = to_tensor(znorm(CU[i : i + chunk], ck["um"], ck["us"]), d)
                V = to_tensor(znorm(VC[i : i + chunk], ck["vm"], ck["vs"]), d)
                C = to_tensor(znorm(CON[i : i + chunk], ck["cm"], ck["cs"]), d)
                out.append((net.V(U, V, C) * ck["ysd"] + ck["ym"]).cpu().numpy())
        return _np.concatenate(out)

    return predict


# ---------------- 단조성 제약 (계약 구조상 확실한 부호를 손실에 넣는다) ----------------
# 쿠폰 ↑ → 가격 ↑ (누적지급률도 같은 비율로) · 행사가 전체 ↑ → 가격 ↓ · KI 배리어 ↑ → 가격 ↓
# 리자드 지급률 ↑ → 가격 ↑ · 리자드 배리어 ↑ → 가격 ↓   (뒤 둘은 리자드 상품만)
# MC 라벨을 더 만들지 않고 기울기·부호를 잡는 것이 목적. 종전에는 scratch/train_mono.py 에만 있었다.
iS, iP, iLB, iLP, iB, iC = slice(0, 12), slice(12, 24), slice(24, 36), slice(36, 48), 48, 49
D_CPN, D_STRK, D_BARR, D_LZB, R_LZP = 0.005, 0.02, 0.02, 0.02, 1.25
LZ_NEUTRAL = 1.2


def _perturb(Craw, kind):
    """원 스케일 계약 배치에 한 가지 변형을 적용한다.

    `module.contract` 의 섭동 규칙과 **같은 규칙의 torch 판**이다 — 규칙을 바꾸면 양쪽을
    같이 고쳐야 한다.

    입력
      Craw : Tensor(b, 51) — 원 스케일 계약 배치. 원본은 바꾸지 않는다
      kind : "coupon" | "strike" | "barrier" | "lz_pmt" | "lz_barr"
    출력
      (X, sign, mask) — X    : Tensor(b, 51), 변형된 계약
                        sign : float, 기대 부호 (+1 이면 가격이 올라야 한다)
                        mask : Tensor(b,), 이 변형이 의미 있는 행만 1
                               (노낙인은 barrier 가 0, 리자드 없는 상품은 리자드 두 종이 0)
    예외
      ValueError — 모르는 kind
    """
    X = Craw.clone()
    if kind == "coupon":
        c = X[:, iC].clamp(min=1e-6)
        r = (c + D_CPN) / c
        X[:, iP] = X[:, iP] * r[:, None]
        X[:, iC] = c + D_CPN
        return X, +1.0, torch.ones_like(c)
    if kind == "strike":
        X[:, iS] = X[:, iS] + D_STRK
        return X, -1.0, torch.ones_like(X[:, iC])
    if kind == "barrier":
        m = (X[:, iB] < 0.99).float()
        X[:, iB] = X[:, iB] + D_BARR * m
        return X, -1.0, m
    lz = (Craw[:, iLB] < LZ_NEUTRAL).float()  # 회차별 리자드 유무
    m = (lz.sum(1) > 0).float()
    if kind == "lz_pmt":
        X[:, iLP] = X[:, iLP] * (1 + (R_LZP - 1) * lz)
        return X, +1.0, m
    if kind == "lz_barr":
        X[:, iLB] = X[:, iLB] + D_LZB * lz
        return X, -1.0, m
    raise ValueError(kind)


KINDS = ("coupon", "strike", "barrier", "lz_pmt", "lz_barr")


def train_curve_mono(
    D,
    cfg,
    tr,
    te,
    target,
    va=None,
    save_path=None,
    target_ratio=0.1,
    verbose=False,
    kinds=None,
    fixed_w=None,
    margin_krw=0.0,
):
    """train_curve 와 같되 단조성 hinge 페널티를 더한다 — **기준 모형의 `+m`** 이 이것이다.

    MC 라벨을 더 만들지 않고 계약 반응의 부호·기울기를 잡는 것이 목적이다.

    입력
      D, cfg       : 공통 번들과 설정
      tr, te       : ndarray[int] — 학습·예측 행 (te 는 쓰지 않는다. 규약 자리)
      target       : ndarray(n,) — 보통 D.MC
      va           : ndarray[int] | None — 검증 행
      save_path    : str | None — 가중치 + 표준화 통계 + 단조 설정을 저장한다
      target_ratio : float — 자동 가중치 모드에서 데이터손실 대비 목표 비율
      verbose      : bool — 500 스텝마다 손실과 가중치를 찍는다
      kinds        : 반복자[str] | None — 걸 변형. None 이면 KINDS 전부.
                     기준 모형은 ("lz_pmt",) 하나만 쓴다
      fixed_w      : float | None — 단조 가중치를 고정한다.
                     **자동 산정은 ml→0 에서 폭발**하므로 기준 모형은 고정값을 쓴다
      margin_krw   : float — 요구하는 최소 반응 [원]. 0 이면 부호만 본다
    출력
      callable(idx) -> ndarray — 원본 스케일 예측기.
                                 `train_curve` 와 달리 **예측기 하나만** 돌려준다
    """
    dev = D.DEV
    B = cfg["train"]["batch"]
    NIT = cfg["train"]["nit"]
    P = cfg["networks"]["P"]
    torch.manual_seed(cfg["seed"])
    um, us = zstats(D.CURVE, tr)
    vm, vs = zstats(D.VC, tr)
    cm, cs = zstats(D.CON, tr)
    Un = to_tensor(znorm(D.CURVE, um, us), dev)
    Vn = to_tensor(znorm(D.VC, vm, vs), dev)
    Cn = to_tensor(znorm(D.CON, cm, cs), dev)
    cmt, cst = to_tensor(cm, dev), to_tensor(cs, dev)
    net = CurveOperatorV2(Vn.shape[1], Cn.shape[1], P).to(dev)
    opt = _opt(net, cfg)
    ym, ysd = float(target[tr].mean()), float(target[tr].std() + 1e-8)
    Y = to_tensor((target - ym) / ysd, dev)
    ntr = len(tr)
    trt = torch.tensor(tr, device=dev)
    es = _EarlyStop(net, cfg, va, dev)
    w_mono = 0.0
    REW = 50  # REW 스텝마다 가중치 재산출

    KS = tuple(kinds) if kinds else KINDS  # 위반하는 항목에만 걸 수 있게
    mg = margin_krw / 10000.0 / ysd  # 원 단위 최소 반응 → 정규화 단위

    def mono_loss(bb, pr):
        """흔든 입력을 한 번에 forward 해 부호가 틀린 만큼만 벌점을 매긴다.

        KS 의 변형을 모두 쌓아 **한 번의 forward** 로 처리한다 (변형 수만큼 배치가 커진다).

        입력
          bb : Tensor[int] — 배치 행 인덱스
          pr : Tensor(b,) — 그 배치의 기준 예측 (표준화 공간)
        출력
          Tensor() — 마스크가 켜진 칸 수로 나눈 평균 위반량.
                     mg=0 이면 부호만, mg>0 이면 '최소 기울기' 를 요구한다
        """
        Craw = Cn[bb] * cst + cmt
        Xs, sg, ms = [], [], []
        for k in KS:
            x, s, m = _perturb(Craw, k)
            Xs.append(torch.clamp((x - cmt) / cst, -ZCLIP, ZCLIP))
            sg.append(s)
            ms.append(m)
        Xc = torch.cat(Xs, 0)
        nrep = len(KS)
        pr2 = net.V(Un[bb].repeat(nrep, 1), Vn[bb].repeat(nrep, 1), Xc)
        d = (pr2 - pr.repeat(nrep)).view(nrep, -1)
        sg = torch.tensor(sg, device=dev)[:, None]
        ms = torch.stack(ms)
        viol = torch.relu(mg - sg * d) * ms  # mg=0 이면 부호만, mg>0 이면 '최소 기울기' 요구
        return viol.sum() / ms.sum().clamp(min=1.0)

    for it in range(NIT):
        bb = trt[torch.randint(0, ntr, (B,), device=dev)]
        pr = net.V(Un[bb], Vn[bb], Cn[bb])
        dl = ((pr - Y[bb]) ** 2).mean()
        ml = mono_loss(bb, pr)
        if fixed_w is not None:  # 고정 가중치 (자동산정은 ml→0 에서 폭발)
            w_mono = fixed_w
        elif it % REW == 0:
            w_new = float(target_ratio * dl.detach() / (ml.detach() + 1e-8))
            w_mono = min(w_new if it == 0 else 0.7 * w_mono + 0.3 * w_new, 10.0)
        if verbose and it % 500 == 0:
            print(
                f"    it{it:4d} data {float(dl):.5f} mono {float(ml):.5f} w {w_mono:.3g}",
                flush=True,
            )
        loss = dl + w_mono * ml
        opt.zero_grad()
        loss.backward()
        opt.step()
        if es.step(it, lambda vi: ((net.V(Un[vi], Vn[vi], Cn[vi]) - Y[vi]) ** 2).mean()):
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
                "nvc": Vn.shape[1],
                "ncon": Cn.shape[1],
                "um": um,
                "us": us,
                "vm": vm,
                "vs": vs,
                "cm": cm,
                "cs": cs,
                "ym": ym,
                "ysd": ysd,
                "mono_weight": w_mono,
                "target_ratio": target_ratio,
                "kinds": list(KS),
            },
            save_path if save_path.endswith(".pt") else save_path + ".pt",
        )

    def p(idx):
        """학습된 망으로 원본 스케일 예측을 낸다.

        입력
          idx : array-like[int] — 행 인덱스
        출력
          ndarray — 길이 len(idx)
        """
        with torch.no_grad():
            i = torch.tensor(np.asarray(idx), device=dev)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()

    return p
