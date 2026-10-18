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
    return torch.optim.Adam(
        net.parameters(), cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )


class _EarlyStop:
    """val 손실 기반 조기종료 + 최적 가중치 복원. va 비면 비활성."""

    def __init__(self, net, cfg, va, dev):
        self.on = va is not None and len(va) > 0
        self.net = net
        self.every = cfg["train"].get("es_every", 100)
        self.patience = cfg["train"].get("es_patience", 6)
        self.vat = torch.tensor(np.asarray(va), device=dev) if self.on else None
        self.best = float("inf")
        self.best_state = None
        self.wait = 0

    def step(self, it, val_loss_fn):
        """(it+1) 마다 호출; 개선 없으면 True(=중단) 반환."""
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
        if self.on and self.best_state is not None:
            self.net.load_state_dict(self.best_state)


# ===== DeepONet-Curve (데이터기반, stage-1 앵커) =====
def _loss_fn(loss):
    """정규화 스케일 손실: mse(기본) | l1(MAE) | huber. (mape 는 train_curve 에서 원본 스케일로 별도 처리)"""
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
    """target=D.MC → 하이브리드 앵커(이론가) / target=D.FAIR → 직접.
    loss: 'mse'(기본)|'l1'|'huber'|'mape'. mape 는 원본 스케일 상대오차. va → val 조기종료.
    moe_k>0 이면 CurveOperatorMoE(K=moe_k, 배리어게이트 ib=moe_ib) 사용 (arm A4b)."""
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

    def bloss(pr, idx):  # 배치 손실 (mape 는 역정규화한 원본 스케일 상대오차)
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
        with torch.no_grad():
            i = torch.tensor(idx, device=dev)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()

    return (p(tr), p(te), p) if return_predict else (p(tr), p(te))


def load_curve_predictor(D, path):
    """저장된 DeepONet-Curve 가중치를 로드해 predict(idx)->np.array 반환 (forward만; 재학습 없음)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CurveOperatorV2(ck["nvc"], ck["ncon"], ck["P"]).to(D.DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    Un = to_tensor(znorm(D.CURVE, ck["um"], ck["us"]), D.DEV)
    Vn = to_tensor(znorm(D.VC, ck["vm"], ck["vs"]), D.DEV)
    Cn = to_tensor(znorm(D.CON, ck["cm"], ck["cs"]), D.DEV)
    ym, ysd = ck["ym"], ck["ysd"]

    def predict(idx):
        with torch.no_grad():
            i = torch.tensor(np.asarray(idx), device=D.DEV)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()

    return predict


def load_curve_forward(path, dev=None, chunk=16384):
    """저장 가중치 -> **임의 입력** 예측기 predict(CURVE, VC, CON) -> np.ndarray.

    load_curve_predictor 는 D 를 닫아 잡고 행 인덱스만 받으므로 '계약을 바꾼 입력' 을 표현할 수 없다.
    계약조건 민감도처럼 CON 을 직접 만들어 넣어야 하는 쓰임은 이쪽을 쓴다.
    정규화 통계는 체크포인트에 든 **학습 폴드 기준** 값을 그대로 적용한다."""
    import numpy as _np

    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    net = CurveOperatorV2(ck["nvc"], ck["ncon"], ck["P"]).to(d)
    net.load_state_dict(ck["state"])
    net.eval()

    def predict(CU, VC, CON):
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
    """원 스케일 계약 배치에 한 가지 변형 → (변형된 계약, 기대 부호, 적용 마스크)."""
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
    """train_curve 와 같되 단조성 hinge 페널티를 추가. 반환: predict(idx)."""
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
        """흔든 입력을 한 번에 forward → 부호가 틀린 만큼만 벌점."""
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
        with torch.no_grad():
            i = torch.tensor(np.asarray(idx), device=dev)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()

    return p
