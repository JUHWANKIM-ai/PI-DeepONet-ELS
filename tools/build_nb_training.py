# -*- coding: utf-8 -*-
"""2_model_training.ipynb 생성 — module 을 불러와 **실제로 학습하는** 노트북.

 인계용이므로 결과 CSV 를 읽어 보여주는 것이 아니라, 노트북 안에서 번들을 만들고
 가중치를 학습·저장하고 OOS 지표까지 낸다. 기본값은 기존 가중치가 있으면 건너뛴다
 (RETRAIN=True 로 강제 재학습).

 실행: python tools/build_nb_training.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "2_model_training.ipynb"


def _src(t):
    return [l + "\n" for l in t.split("\n")[:-1]] + [t.split("\n")[-1]]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(t.strip())}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _src(t.strip("\n"))}


cells = []

cells.append(md("""
# 2. 모형 학습

`module` 패키지를 불러와 **이 노트북 안에서 직접 학습**한다. 저장된 결과를 읽어 보여주는 것이 아니라
번들 구성 → 학습 → 가중치 저장 → OOS 평가까지 한 번에 돈다.

| | |
|---|---|
| 기준 모형 | DeepONet-Curve (Direct 타깃 + 단조손실) |
| 타깃 | MC 이론가, 현행 방법론 (1장 참고) |
| 학습 풀 | 기초자산 1·2·3개 **통합** |
| 검증 | 확장 walk-forward 4폴드 × 5시드, 시험구간은 항상 미래 |
| 소요 | 폴드당 약 11초 → 4폴드 × 5시드 약 4분 (CUDA) |

**왜 통합 학습인가.** 3스타만 학습한 모형으로 1스타를 예측하면 무너진다 (아래 2-4절에서 직접 확인).
슬롯 중복 패딩으로 1·2스타를 3자산 입력 규약에 수용한다 — 1스타는 A,A,A / 2스타는 A,B,B,
중복쌍 상관은 정확히 1.0.

**누수 방지는 구조로 보장한다.** 폴드는 확장 윈도우이고 시험구간은 언제나 미래다. 검증표본은
각 폴드의 학습구간 안에서만 뽑고, 정규화 통계는 학습 인덱스로만 계산한다.
"""))

cells.append(code("""
import copy, time, yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from util import file_manager as fm
from module import star as ST
from module import train as TR

SEEDS = range(5)                 # 시드 수. 빠르게 확인하려면 range(1)
RETRAIN = False                  # True 면 기존 가중치를 무시하고 다시 학습
MONO = dict(fixed_w=1.0, kinds=["lz_pmt"], margin_krw=8.0)   # 단조손실 설정
FACE = 10000
MDIR = fm.RESULT / "models"; MDIR.mkdir(parents=True, exist_ok=True)

cfg0 = yaml.safe_load(open(fm.ROOT / "config.yaml", encoding="utf-8"))
print("장치", "cuda" if torch.cuda.is_available() else "cpu",
      "| 반복", cfg0["train"]["nit"], "| 배치", cfg0["train"]["batch"],
      "| 잠재차원", cfg0["networks"]["P"])
"""))

cells.append(md("""
## 2-1. 학습 번들

`module.star.make_all()` 이 3스타(`module.mcvar`)와 1·2스타를 같은 입력 규약으로 합치고
발행일로 정렬한 뒤 폴드를 끊는다.

입력은 세 갈래다 — **곡선**(부트스트랩 제로금리 10노드) · **변동성·상관**(6개월 EWMA 3개 +
상관 3개 + 실효변동성 + 배당수익률 3개) · **계약**(행사가·지급률·리자드 각 12칸 + 배리어·쿠폰·만기).
"""))

cells.append(code("""
t0 = time.time()
D = ST.make_all()
oos = np.concatenate([te for _, _, te in D.WF])
star_n = {int(k): int(v) for k, v in zip(*np.unique(D.nstar, return_counts=True))}
print("번들 {:,}건 ({:.0f}s) | 기초자산 수별 {}".format(D.n, time.time() - t0, star_n))
print("  곡선 {} | 변동성·상관 {} | 계약 {}".format(D.CURVE.shape, D.VC.shape, D.CON.shape))
print("  폴드 {}개 | OOS 합계 {:,}건".format(len(D.WF), len(oos)))
for k, (tr, va, te) in enumerate(D.WF):
    print("    폴드{} 학습 {:6,} 검증 {:5,} 시험 {:6,}".format(k, len(tr), len(va), len(te)))
"""))

cells.append(md("""
## 2-2. V_base 프록시 (잔차 학습용, 선택)

논문의 준해석적 베이스 프록시를 우리 ELS 에 이식한 것이다 (`module.vbase`). 다자산 worst-of 를
스칼라 GBM 하나로 바꾸되, 원문의 단순 평균 변동성 대신 **Clark(1961) worst-of 적률맞춤**을 쓴다.

기준 모형(Direct)은 이걸 쓰지 않는다. 잔차 타깃 변형과 비교하려고 함께 계산해 둔다.
""" ))

cells.append(code("""
vb = ST.vbase_all(D)
y = D.MC[oos] * FACE
e = (vb[oos] - D.MC[oos]) * FACE
print("V_base 단독 (OOS {:,}건)".format(len(oos)))
print("  MAE {:.0f}원 | 편향 {:+.0f}원 | 상관 {:.4f} | 분산설명 {:.1f}%".format(
    np.abs(e).mean(), e.mean(), np.corrcoef(vb[oos], D.MC[oos])[0, 1],
    (1 - ((D.MC[oos] - vb[oos]).std() / D.MC[oos].std()) ** 2) * 100))
print("  → 프록시의 역할은 '맞히는 것' 이 아니라 '구조를 떼어내는 것' 이다.")
"""))

cells.append(md("""
## 2-3. 학습

`module.train.train_curve_mono` 가 데이터손실(MSE)에 **단조성 hinge** 를 더해 학습한다.
계약 구조상 부호가 확실한 다섯 방향을 배치마다 흔들어, 부호가 틀린 만큼만 벌점을 준다.

| 흔드는 것 | 기대 부호 |
|---|---|
| 쿠폰 ↑ (누적지급률도 같은 비율) | 가격 ↑ |
| 행사가 전체 ↑ | 가격 ↓ |
| KI 배리어 ↑ (낙인 상품만) | 가격 ↓ |
| 리자드 지급률 ↑ (리자드 상품만) | 가격 ↑ |
| 리자드 배리어 ↑ (리자드 상품만) | 가격 ↓ |

기본 설정은 `kinds=["lz_pmt"]` — 기준 모형이 유일하게 위반하던 항목만 건다.
MC 라벨을 더 만들지 않고 기울기·부호를 잡는 것이 목적이다.
"""))

cells.append(code("""
def train_variant(tag, resid=False, feat=False, only3=False, seeds=SEEDS):
    \"\"\"가중치를 학습·저장한다. 이미 있으면 건너뛴다 (RETRAIN=True 면 다시 학습).\"\"\"
    Dx = copy.copy(D)
    CONf = np.concatenate([D.CON, vb[:, None]], 1).astype("float32")
    Dx.CON = CONf if feat else D.CON
    Dx.MC = (D.MC - vb).astype("float32") if resid else D.MC.astype("float32")
    is3 = D.nstar == 3
    made, t0 = 0, time.time()
    for s in seeds:
        cfg = copy.deepcopy(cfg0); cfg["seed"] = s
        for k, (tr, va, te) in enumerate(D.WF):
            base = str(MDIR / "{}_deeponet_anchor_seed{}_fold{}".format(tag, s, k))
            if Path(base + ".pt").exists() and not RETRAIN:
                continue
            TR.train_curve_mono(Dx, cfg, tr[is3[tr]] if only3 else tr, te, Dx.MC,
                                va=va[is3[va]] if only3 else va, save_path=base, **MONO)
            made += 1
    print("  [{}] 신규 {}개 ({:.1f}분)".format(tag, made, (time.time() - t0) / 60))
    return tag


# 기준 모형 + 비교군 (학습 풀만 바꾼 3스타 전용)
train_variant("ball_d", resid=False, feat=False, only3=False)
train_variant("ball3_d", resid=False, feat=False, only3=True)
print("저장 위치:", MDIR)
"""))

cells.append(md("""
## 2-4. OOS 평가

저장된 가중치를 되읽어 시험구간을 예측한다. `module.train.load_curve_forward` 는 `D` 를 닫아 잡지
않고 배열을 직접 받으므로, 3절에서 계약을 바꿔 넣을 때도 같은 함수를 쓴다.

**1단계 성능은 MC 재현만으로 잰다.** 공정가 지표는 섞지 않는다.
"""))

cells.append(code("""
def predict(tag, feat=False, resid=False):
    \"\"\"폴드별 가중치로 OOS 전량을 예측 (시드 평균).\"\"\"
    CONf = np.concatenate([D.CON, vb[:, None]], 1).astype("float32")
    fold = np.full(D.n, -1, int)
    for k, (tr, va, te) in enumerate(D.WF):
        fold[te] = k
    out = np.full((len(SEEDS), D.n), np.nan)
    for si, s in enumerate(SEEDS):
        for k in range(len(D.WF)):
            i = np.where(fold == k)[0]
            f = TR.load_curve_forward(str(MDIR / "{}_deeponet_anchor_seed{}_fold{}".format(tag, s, k)) + ".pt")
            v = f(D.CURVE[i], D.VC[i], (CONf if feat else D.CON)[i].astype("float32"))
            out[si, i] = (v + (vb[i] if resid else 0.0)) * FACE
    return out, fold


def metrics(pred, idx):
    y = D.MC[idx] * FACE
    r = []
    for si in range(pred.shape[0]):
        e = pred[si, idx] - y
        r.append(dict(R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                      MAPE=(np.abs(e) / np.abs(y)).mean() * 100, MAE=np.abs(e).mean(),
                      RMSE=float(np.sqrt((e ** 2).mean())), MSE=float((e ** 2).mean()),
                      bias=e.mean()))
    return {k: (float(np.mean([x[k] for x in r])),
                float(np.std([x[k] for x in r], ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0)
            for k in r[0]}


P_all, fold = predict("ball_d")
P_3, _ = predict("ball3_d")
np.save(fm.CACHE / "nb_pred_ball_d.npy", P_all)        # 3장 시각화가 되읽는다
np.save(fm.CACHE / "nb_pred_ball3_d.npy", P_3)

rows = []
for gl, gm in [("1star", D.nstar == 1), ("2star", D.nstar == 2),
               ("3star", D.nstar == 3), ("ALL", np.ones(D.n, bool))]:
    idx = oos[gm[oos]]
    for lab, Pm in [("통합 학습 (1·2·3)", P_all), ("3스타 전용", P_3)]:
        m = metrics(Pm, idx)
        rows.append(dict(group=gl, model=lab, n=len(idx), mc_sd=(D.MC[idx] * FACE).std(),
                         **{k: v[0] for k, v in m.items()},
                         **{k + "_se": v[1] for k, v in m.items()}))
T = pd.DataFrame(rows)
T.to_csv(fm.stat("nb_training_level"), index=False, encoding="utf-8-sig")
print(T[["group", "model", "n", "R2", "R2_se", "MAPE", "MAPE_se", "MAE", "RMSE", "MSE", "bias"]]
      .round(4).to_string(index=False))
"""))

cells.append(md("""
## 결론

1. **통합 학습이 필수다.** 3스타 전용 모형은 1스타에서 무너진다. 통합해도 3스타 성능 손실은 거의 없다.
2. **단조손실은 라벨을 늘리지 않고 기울기를 잡는다.** MC 를 더 돌리지 않으므로 비용이 들지 않는다.
3. **가중치는 `result/models/` 에 폴드·시드별로 저장된다.** 3장 시각화와 계약조건 분석이 이걸 되읽는다.

다음 장(3. 시각화)에서 예측 분포 · 계약조건 반응 · 기초자산별 오차를 그림으로 본다.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "els-abacus", "language": "python",
                                  "name": "els-abacus"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT.name, "| cells", len(cells))
