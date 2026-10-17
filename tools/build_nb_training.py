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

`module` 패키지를 불러와 **이 노트북 안에서 직접 학습**합니다. 저장된 결과를 읽어 보여주는 것이 아니라
번들 구성 → 학습 → 가중치 저장 → OOS 평가까지 한 번에 돕니다.

> **처음 보신다면**: 1장이 만든 MC 가격을 **정답지로 삼아** 신경망에게 "이 시장상태와 이 계약조건이면
> MC 는 얼마를 내놓는다" 를 가르치는 단계입니다.
>
> 입력은 세 덩어리입니다. **금리곡선**(10개 만기의 제로금리), **시장상태**(기초자산 3개의 변동성 ·
> 쌍별 상관 · 배당수익률), **계약조건**(행사가 12칸 · 지급률 12칸 · 리자드 24칸 · KI 배리어 · 쿠폰 · 만기).
> 출력은 가격 하나입니다.

| | |
|---|---|
| 기준 모형 | DeepONet-Curve (Direct 타깃 + 단조손실) |
| 타깃 | MC 이론가, 현행 방법론 (1장 참고) |
| 학습 풀 | 기초자산 1·2·3개 **통합** |
| 검증 | 확장 walk-forward 4폴드 × 5시드, 시험구간은 항상 미래 |
| 소요 | 폴드당 약 11초 → 모형 하나당 약 4분, 두 벌 합쳐 약 6분 (CUDA) |

**왜 통합 학습인가.** 3스타만 학습한 모형으로 1스타를 예측하면 무너집니다(아래 2-4절에서 직접 확인).
그래서 슬롯 중복 패딩으로 1·2스타를 3자산 입력 규약에 수용합니다.

**누수 방지는 구조로 보장합니다.** 폴드는 확장 윈도우이고 시험구간은 언제나 미래입니다.
검증표본은 각 폴드의 학습구간 안에서만 뽑고, 정규화 통계도 학습 인덱스로만 계산합니다.
"""))

cells.append(code("""
import copy, json, os, time, yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from util import file_manager as fm
from module import star as ST
from module import train as TR

SEEDS = range(5)                 # 시드 수. 빠르게 확인하려면 range(1)
MONO = dict(fixed_w=1.0, kinds=["lz_pmt"], margin_krw=8.0)   # 단조손실 설정
FACE = 10000
MDIR = fm.RESULT / "models"; MDIR.mkdir(parents=True, exist_ok=True)

# 라벨이 바뀌면 저장된 가중치는 낡는다. "파일이 있는가" 로 판단하면 낡은 가중치를 조용히
# 재사용하므로, 라벨의 **출처 스탬프**와 가중치가 학습된 스탬프를 비교해 자동으로 결정한다.
# RETRAIN=1 로 강제할 수도 있다.
from tools import build_mc_variants as BMV
WSTAMP = MDIR / ".trained_label_stamp.json"


def weights_stale():
    \"\"\"지금 라벨로 다시 학습해야 하는가.\"\"\"
    want = BMV.read_stamp(BMV.OUT / BMV.OUT_NAME)
    if want is None:
        return False
    if not WSTAMP.exists():
        return any(MDIR.glob("*"))        # 가중치는 있는데 어떤 라벨로 학습했는지 모른다
    got = json.loads(WSTAMP.read_text(encoding="utf-8"))
    return any(str(got.get(k)) != str(want.get(k))
               for k in ("npath", "div", "px_source", "n_rows"))


RETRAIN = os.environ.get("RETRAIN", "0") == "1" or weights_stale()
print("재학습:", RETRAIN, "| 라벨 스탬프", BMV.read_stamp(BMV.OUT / BMV.OUT_NAME))

cfg0 = yaml.safe_load(open(fm.ROOT / "config.yaml", encoding="utf-8"))
print("장치", "cuda" if torch.cuda.is_available() else "cpu",
      "| 반복", cfg0["train"]["nit"], "| 배치", cfg0["train"]["batch"],
      "| 잠재차원", cfg0["networks"]["P"])
"""))

cells.append(md("""
## 2-1. 학습 번들 — 1·2·3스타를 한 배열로

흩어진 자료를 **하나의 입력 규약**으로 모읍니다. 3스타는 그대로, 1·2스타는 슬롯 중복 패딩으로
3자산 형태에 맞춥니다. 그다음 발행일로 다시 정렬하고 walk-forward 폴드를 끊습니다.

아래 출력의 `폴드k 학습/검증/시험` 건수를 보면 **학습구간이 폴드마다 늘어나고(확장 윈도우)
시험구간은 항상 그 뒤**라는 것을 확인할 수 있습니다.

`module.star.make_all()` 이 3스타(`module.mcvar`)와 1·2스타를 같은 입력 규약으로 합친 뒤,
발행일로 정렬하고 폴드를 끊습니다.

입력은 세 갈래입니다. **곡선**(부트스트랩 제로금리 10노드) · **변동성·상관**(6개월 EWMA 3개 +
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

**V_base 란.** worst-of 바스켓을 단일 자산처럼 근사해 닫힌식으로 계산한 **간이 가격**입니다
(Clark 1961 의 적률정합). 정확하지 않지만 "배리어가 깊으면 싸다" 같은 **큰 구조**는 담고 있습니다.

**왜 두는가.** 신경망에게 가격을 통째로 외우게 하는 대신, 간이식이 설명하는 부분을 떼어내고
**나머지 차이만** 학습시키면 더 쉬워질 수 있습니다(Residual 방식). 아래 출력은 V_base 단독 성능인데,
그 역할은 '맞히는 것' 이 아니라 '구조를 떼어내는 것' 이라 MAE 가 커도 문제가 아닙니다.

기준 모형(Direct)은 V_base 를 **쓰지 않습니다.** 비교군으로만 둡니다.

논문의 준해석적 베이스 프록시를 이 ELS 에 이식한 것입니다(`module.vbase`). 다자산 worst-of 를
스칼라 GBM 하나로 바꾸되, 원문의 단순 평균 변동성 대신 **Clark(1961) worst-of 적률맞춤**을 씁니다.

기준 모형(Direct)은 이것을 쓰지 않습니다. 잔차 타깃 변형과 비교하려고 함께 계산해 둘 뿐입니다.
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
## 2-3. 학습 — 일곱 벌을 같은 조건에서

**기준 모형은 `ball_d` 입니다**: 1·2·3스타 **통합** 학습 · Direct 타깃 · 단조손실.
나머지 여섯 벌은 한 번에 한 가지만 바꾼 대조군이고, 노트북 3 의 비교 그림에서 막대 하나씩을
차지합니다. 노트북 3 은 가중치를 **불러올 뿐 학습하지 않으므로**, 여기서 일곱 벌이 다 나와야
그림이 완성됩니다.

| 접두사 | 바꾼 것 | 역할 |
|---|---|---|
| `ball_d` | — | **기준** (Direct + 단조손실, 통합 학습) |
| `ball3_d` | 학습 풀 3스타만 | 통합 학습의 기여를 분리 |
| `ball_dm` | V_base 를 피처로 추가 | 프록시를 피처로 주면 나아지나 |
| `ball_fm` | 타깃을 잔차로 + 피처 추가 | 잔차학습 대조 |
| `ball3_fm` | 위 + 3스타만 | 잔차학습의 학습 풀 효과 |
| `ball_xgb` | 아키텍처 XGBoost | 표형 벤치마크 (같은 입력) |
| `ball3_xgb` | 위 + 3스타만 | |

XGBoost 두 벌은 `.pkl`, DeepONet 다섯 벌은 `.pt` 로 저장됩니다 — 5벌 × 20 + 2벌 × 20 = **140개**.

**단조손실이 하는 일.** 리자드 지급률을 1.25배로 올린 가짜 입력을 함께 넣어보고,
가격이 **내려가면 벌점**을 줍니다. 금융 상식상 뒤집히면 안 되는 방향을 손실로 강제하는 장치입니다.

시드 5개 × 폴드 4개 = 모형당 20개 가중치를 저장합니다. 시드를 여럿 쓰는 이유는
**한 시드의 결과로 순위를 말할 수 없기** 때문입니다. 아래 평가에서 평균과 표준오차를 함께 냅니다.
이미 저장된 가중치가 있으면 건너뜁니다(`RETRAIN = True` 로 다시 학습).

`module.train.train_curve_mono` 가 데이터손실(MSE)에 **단조성 hinge** 를 더해 학습합니다.
계약 구조상 부호가 확실한 다섯 방향을 배치마다 흔들어, 부호가 틀린 만큼만 벌점을 줍니다.

| 흔드는 것 | 기대 부호 |
|---|---|
| 쿠폰 ↑ (누적지급률도 같은 비율) | 가격 ↑ |
| 행사가 전체 ↑ | 가격 ↓ |
| KI 배리어 ↑ (낙인 상품만) | 가격 ↓ |
| 리자드 지급률 ↑ (리자드 상품만) | 가격 ↑ |
| 리자드 배리어 ↑ (리자드 상품만) | 가격 ↓ |

기본 설정은 `kinds=["lz_pmt"]` 입니다. 기준 모형이 유일하게 위반하던 항목만 겁니다.
MC 라벨을 더 만들지 않고도 기울기와 부호를 잡는 것이 목적입니다.
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


def train_xgb(tag, only3=False, seeds=SEEDS):
    \"\"\"XGBoost 앵커를 학습·저장한다 (.pkl). 입력은 DeepONet 앵커와 같은 [곡선|vol·corr|계약].\"\"\"
    from model.benchmark import _xgb_anchor
    is3 = D.nstar == 3
    made, t0 = 0, time.time()
    for s in seeds:
        cfg = copy.deepcopy(cfg0); cfg["seed"] = s
        for k, (tr, va, te) in enumerate(D.WF):
            base = str(MDIR / "{}_anchor_seed{}_fold{}".format(tag, s, k))
            if Path(base + ".pkl").exists() and not RETRAIN:
                continue
            _xgb_anchor(D, cfg, tr[is3[tr]] if only3 else tr,
                        va[is3[va]] if only3 else va, te, save_path=base)
            made += 1
    print("  [{}] 신규 {}개 ({:.1f}분)".format(tag, made, (time.time() - t0) / 60))
    return tag


# 일곱 벌 — 노트북 3 의 eval_all_stars / eval_cross_section 가 이 접두사들을 읽는다.
# 하나라도 빠지면 그림에서 막대가 조용히 사라지므로 목록을 여기서 못박는다.
train_variant("ball_d",   resid=False, feat=False, only3=False)   # 기준
train_variant("ball3_d",  resid=False, feat=False, only3=True)
train_variant("ball_dm",  resid=False, feat=True,  only3=False)
train_variant("ball_fm",  resid=True,  feat=True,  only3=False)
train_variant("ball3_fm", resid=True,  feat=True,  only3=True)
train_xgb("ball_xgb",  only3=False)
train_xgb("ball3_xgb", only3=True)
WSTAMP.write_text(json.dumps(BMV.read_stamp(BMV.OUT / BMV.OUT_NAME) or {}, ensure_ascii=False),
                  encoding="utf-8")        # 어떤 라벨로 학습했는지 남긴다
print("저장 위치:", MDIR, "| 가중치", len(list(MDIR.glob("*"))), "개")
"""))

cells.append(md("""
## 2-4. OOS 평가 — 시험구간 전량

폴드별 가중치로 그 폴드의 **시험구간만** 예측해 이어붙입니다. 학습에 쓰인 구간은 한 건도 들어가지 않습니다.

**표 읽는 법.** `R2` 는 1에 가까울수록, `MAPE`(평균 절대 백분율 오차)는 0에 가까울수록 좋습니다.
`_se` 는 시드 5개의 표준오차로, **두 모형의 차이가 SE 보다 작으면 우열을 말할 수 없습니다.**
`bias` 는 부호 있는 평균 오차입니다. 양수면 모형이 MC 보다 비싸게 부른다는 뜻입니다.

`1star` 행에서 두 모형의 차이가 가장 큽니다. 3스타만 학습한 모형은 1스타에서 무너집니다.

저장된 가중치를 되읽어 시험구간을 예측합니다. `module.train.load_curve_forward` 는 `D` 를 닫아 잡지
않고 배열을 직접 받으므로, 계약을 바꿔 넣는 분석에서도 같은 함수를 씁니다.

**1단계 성능은 MC 재현만으로 잽니다.** 공정가 지표는 섞지 않습니다.
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

1. **통합 학습이 필수입니다.** 3스타 전용 모형은 1스타에서 무너집니다. 통합해도 3스타 성능 손실은 거의 없습니다.
2. **단조손실은 라벨을 늘리지 않고 기울기를 잡습니다.** MC 를 더 돌리지 않으므로 추가 비용이 없습니다.
3. **가중치는 `result/models/` 에 폴드·시드별로 저장됩니다.** 3장 시각화와 계약조건 분석이 이것을 되읽습니다.

다음 장(3. 시각화)에서 예측 분포 · 계약조건 반응 · 기초자산별 오차를 그림으로 봅니다.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "els-abacus", "language": "python",
                                  "name": "els-abacus"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT.name, "| cells", len(cells))
