# -*- coding: utf-8 -*-
"""0_data_preprocessing.ipynb 생성 — 원천 → 학습 데이터셋.

 흡수: 0_data.ipynb + EDA_raw_filter.ipynb
 실행: python generate/build_nb_data.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "0_data_preprocessing.ipynb"


def _src(t):
    return [l + "\n" for l in t.split("\n")[:-1]] + [t.split("\n")[-1]]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(t.strip())}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _src(t.strip("\n"))}


cells = []

cells.append(md("""
# 0. 데이터 전처리

> **이 노트북을 처음 보신다면 아래 세 절(프로젝트 개요 · 읽는 순서 · 용어)을 먼저 읽어 주세요.**
> 네 장 중 어디서부터 봐도 되지만, 실행은 0 → 1 → 2 → 3 순서여야 합니다.

---

## 이 프로젝트가 푸는 문제

ELS 한 건의 **이론가**를 몬테카를로(MC)로 계산하는 데 상품당 약 **66밀리초**가 걸립니다
(방법론 경로 수 10만 기준. 4만 경로로 줄이면 28밀리초이지만 잡음이 1.7배로 커집니다).
전체 58,913건을 한 번 다 돌리면 **약 두 시간 반**입니다. 계약조건을 바꿔가며 즉시 견적을 내거나, 장중에 북 전체를 다시 평가하기에는 느립니다.

이 프로젝트는 그 MC 프라이서를 **신경망 대리모형(surrogate model)으로 근사**합니다.
시장상태와 계약조건을 넣으면 MC가 내놓았을 가격을 곧바로 돌려주는 모형입니다.

| | MC 프라이서 | 대리모형 |
|---|---|---|
| 상품당 시간 | 66 ms (10만 경로) | **1.9 µs** (전량 일괄) |
| 정확도 | 기준 (참값) | R² 0.948 · MAPE 0.55% |

**중요한 전제.** 대리모형이 맞히려는 것은 **시장가격이 아니라 MC 이론가**입니다.
"MC가 내놓을 답을 MC보다 빨리" 가 목표이지, "시장보다 잘 맞히기" 가 목표가 아닙니다.
그래서 모든 성능 지표는 *MC 대비* 로 잽니다.

---

## 네 장의 노트북이 하는 일

| | 노트북 | 하는 일 | 만드는 것 | 비용 |
|---|---|---|---|---|
| **0** | 데이터 전처리 | 원천 공시 → 학습 데이터셋 | `els3_dataset.parquet` · `els12_dataset.parquet` · `ml.csv` · `deeponet.csv` | 수 분 |
| **1** | MC 프라이서 | 현행 방법론 정의 + 재현 검증 + **MC 라벨 산출** | `mc_variants*.parquet` (정답지) | 약 70분 |
| **2** | 모형 학습 | 대리모형 학습 + OOS 평가 | `result/models/` 가중치 40개 | 약 6분 |
| **3** | 시각화 | 그림 23장 + 표 13개 | `result/image/` · `result/statistics/` | 수 분 |

각 노트북은 **이미 만들어둔 산출물이 있으면 건너뜁니다.** 처음부터 다시 만들려면 맨 위 설정 셀의
`REBUILD` / `RETRAIN` 을 `True` 로 바꾸세요.

---

## 용어

**상품 구조**

| 용어 | 뜻 |
|---|---|
| **ELS** | 주가연계증권. 기초자산 가격에 따라 상환금이 정해지는 파생결합증권 |
| **스텝다운(STEP)** | 조기상환 행사가가 회차마다 낮아지는 구조. 6개월마다 평가해 전 기초자산이 행사가 이상이면 쿠폰과 함께 조기상환 |
| **worst-of** | 기초자산 중 **가장 못한 것** 기준으로 판정. 하나라도 빠지면 상환 안 됨 |
| **KI 배리어(낙인)** | 만기까지 한 번이라도 이 수준 아래로 떨어지면 원금손실 구간에 들어가는 선. `B = 0.5` 면 50% |
| **노낙인** | KI 배리어가 없는 구조. 이 데이터셋에서는 `B = 1.0` 으로 인코딩합니다 — 만기 미상환이면 항상 worst 를 수취한다는 뜻이고, MC 엔진이 `B >= 1.0` 을 '항상 낙인' 으로 처리합니다 |
| **리자드(LIZARD)** | 특정 회차에 한해 더 낮은 배리어를 두고, 그 위면 축소된 쿠폰으로 조기상환시키는 특약 |
| **1·2·3스타** | 기초자산 **개수**. 3스타 = 기초자산 3개. 이 저장소에서 '스타' 는 등급이 아니라 개수입니다 |

**가격**

| 용어 | 뜻 |
|---|---|
| **공정가(fair)** | 증권사가 공시한 평가금액. 액면 10,000원 대비 비율로 다룹니다 |
| **이론가(MC)** | 현행 방법론으로 직접 산출한 몬테카를로 가격. **이 프로젝트의 정답지** |
| **격차(fair − MC)** | 공정가와 이론가의 차이. 마진·모형차이·자료차이가 섞여 있어 '어느 쪽이 맞다' 를 가릴 수 없습니다 |

**모형**

| 용어 | 뜻 |
|---|---|
| **대리모형 / surrogate** | 느린 계산(MC)을 흉내 내는 빠른 근사 모형 |
| **DeepONet** | 함수를 입력으로 받는 연산자 학습 신경망. 여기서는 **금리 곡선 전체**를 입력으로 받기 위해 씁니다 (곡선을 숫자 몇 개로 요약하지 않음) |
| **Direct / Residual** | 타깃을 MC 가격 **그대로** 학습하면 Direct, 간이 근사식(V_base)과의 **차이만** 학습하면 Residual |
| **단조손실(+m)** | "리자드 지급률이 오르면 가격도 올라야 한다" 처럼 **방향이 뒤집히면 벌점**을 주는 추가 손실항 |
| **walk-forward** | 과거로 학습 → 미래로 시험을 시점을 밀어가며 반복하는 검증. 시험구간은 **언제나 미래** |
| **OOS** | out-of-sample. 학습에 쓰지 않은 시험구간 |
| **슬롯 중복 패딩** | 기초자산이 1~2개인 상품을 3자산 입력 규약에 맞추는 방법. 1스타는 (A,A,A), 2스타는 (A,B,B) 로 채우고 중복쌍 상관은 1.0. worst-of 는 최솟값이라 중복 슬롯은 값에 영향이 없습니다 |

**인과성.** 모든 입력은 발행일 **이전** 자료만 씁니다. 미래 정보를 쓰면 성능이 좋아 보여도 실제로는 못 씁니다.

---

## 이 노트북(0장)이 하는 일

DART 원천 공시 → 학습 데이터셋. `module.build_source` 를 불러와 이 노트북 안에서 직접 빌드합니다.

| 단계 | 산출 |
|---|---|
| 원천 CSV 3종 + 가격·커브 캐시 | `data/els3_dataset.parquet` (기초자산 3개) |
| MC 열 + 모델링 데이터셋 | `data/ml.csv` · `data/deeponet.csv` |
| 1·2스타 확장 (슬롯 중복 패딩) | `data/els12_dataset.parquet` |

**유니버스 정의**: 기초자산 3개 · KRW · 4구조(STEP/LIZARD × 낙인/노낙인) ·
공정가비율 0.70~1.05 · 만기 0.5~5.0년. 범위를 벗어난 건은 이상치로 보고 제외합니다
(사유별 탈락 건수는 0-1 절에서 확인).
"""))

cells.append(code("""
import os

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from util import file_manager as fm, plot as _plot      # import 만으로 dpi400·투명·PPT 밴드
from module import build_source as BS
from module.schedule import load_schedules, LZ_BARR_NEUTRAL

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.transparent": False,
                     "axes.grid": True, "grid.color": "#e8e8e8",
                     "axes.spines.top": False, "axes.spines.right": False})
C1, C2 = "#1f6f9c", "#e07a5f"
# 저장본이 지금 코드가 만들 것과 다르면 다시 빌드한다. "파일이 있는가" 로 판단하면
# 시드 정의를 바꿔도 저장본이 남아 라벨 재생성이 헛돈다 — 실제로 그랬다.
# 유니버스가 늘어난 경우는 빌드해 봐야 알므로 REBUILD=1 로 강제한다.
def _stale_saved():
    from module.seeds import seeds_stale
    p_ = fm.DATA / "els3_dataset.parquet"
    if not p_.exists():
        return True
    return seeds_stale(pd.read_parquet(p_, columns=["item", "mc_seed"]))


REBUILD = os.environ.get("REBUILD", "0") == "1" or _stale_saved()
print("재빌드:", REBUILD)
print("원천:", fm.RAW)
"""))

cells.append(md("""
## 0-1. 필터 퍼널 — 원천 몇 건이 왜 빠지는가

원천 공시에는 이 프로젝트가 다룰 수 없는 건이 섞여 있습니다(기초자산 4개 이상, 외화, 스케줄 결측 등).
어떤 조건에서 몇 건이 빠졌는지를 단계별로 쌓아 보여줍니다. **탈락이 특정 시기·특정 구조에 쏠려 있으면
표본이 편향되므로**, 아래 그림에서 그 쏠림을 확인합니다.

원천에서 유니버스까지 몇 단계를 거치며 무엇이 왜 떨어지는지 셉니다.
`build_source` 는 탈락 사유를 `df.attrs["drops"]` 로 돌려줍니다.
"""))

cells.append(code("""
ac = pd.read_csv(fm.RAW / "LAKE_V2_DART_AUTO_CALL.csv", low_memory=False)
ud = pd.read_csv(fm.RAW / "LAKE_V2_DART_UDLY_INFO.csv", low_memory=False)
for c in ("ISU_DT", "MAT_DT"):
    ac[c] = pd.to_datetime(ac[c], errors="coerce")
ac["tenor"] = (ac["MAT_DT"] - ac["ISU_DT"]).dt.days / 365.25
ac["fair"] = ac["FAIR_VALUE"] / ac["ISU_PRC_DETAIL"]
nu = ud.groupby("ITEM_CD")["UDLY_ID"].nunique()
three = nu[nu == 3].index

steps = [("all filings", len(ac))]        # 그림 라벨이라 영문 (그림 글자는 전부 영문 규칙)
m = ac["ITEM_CD"].isin(three); steps.append(("3 underlyings", int(m.sum())))
m &= ac["CUR_CD"].eq("KRW"); steps.append(("+ KRW", int(m.sum())))
st = pd.Series(False, index=ac.index)
for o, k in BS.STRUCTS:
    st |= (ac["OPT_TYPE"].eq(o) & ac["KNCK_IN_YN"].eq(k))
m &= st; steps.append(("+ 4 structures", int(m.sum())))
m &= ac["fair"].between(BS.FAIR_LO, BS.FAIR_HI); steps.append(("+ fair ratio", int(m.sum())))
m &= ac["tenor"].between(BS.TEN_LO, BS.TEN_HI); steps.append(("+ tenor range", int(m.sum())))

F = pd.DataFrame(steps, columns=["step", "n"])
F["kept_%"] = (F.n / F.n.iloc[0] * 100).round(1)
print(F.to_string(index=False))

fig, a = plt.subplots(figsize=(13.2, 3.8))
a.barh(np.arange(len(F))[::-1], F.n, color=C1, edgecolor="white", lw=.8)
for i, (n, v) in enumerate(zip(F.step, F.n)):
    a.text(v, len(F) - 1 - i, "  {:,} ({:.0f}%)".format(v, v / F.n.iloc[0] * 100),
           va="center", fontsize=9.5)
a.set_yticks(np.arange(len(F))[::-1]); a.set_yticklabels(F.step)
a.set_xlabel("products"); a.margins(x=.22)
a.set_title("Universe filter funnel", fontweight="bold")
fig.tight_layout(); fig.savefig(fm.image("data_funnel"), dpi=200, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""
## 0-2. 데이터셋 빌드 — 공시를 고정폭 배열로

상품마다 조기상환 회차 수가 다릅니다(4~12회). 신경망은 가변 길이를 그대로 못 받으므로
**48칸 고정폭**으로 정규화합니다. 행사가·지급률·리자드배리어·리자드지급률 각 12칸입니다.
회차가 모자라는 칸과 리자드가 없는 상품은 **중립값**으로 채웁니다
(리자드 배리어 1.2 = 절대 발동 안 함, 지급률 0).

`build_source()` 가 후보를 돌며 시장입력(변동성·상관·커브)을 붙이고, 스케줄을 48칸 고정폭으로
정규화합니다. `save=False` 로 돌리면 기존 parquet 를 덮어쓰지 않습니다.
"""))

cells.append(code("""
if REBUILD:
    df = BS.build_source(save=True, verbose=True)
    drops = df.attrs.get("drops", {})
else:
    df = pd.read_parquet(fm.DATA / "els3_dataset.parquet")
    drops = {}
    print("저장본을 읽음 (REBUILD=True 로 다시 빌드)")
print("데이터셋 {:,}건 × {}열".format(len(df), df.shape[1]))
if drops:
    print("탈락 사유:", dict(sorted(drops.items(), key=lambda x: -x[1])))
print()
print("구조별:", df.groupby(["opt_type", "ki_yn"]).size().to_dict())
"""))

cells.append(md("""
### MC 이론가 열과 모델링 데이터셋

`build_source` 가 만드는 parquet 에는 아직 MC 가격이 없습니다. 뒤 단계가 쓰려면 두 가지를 더 붙여야 합니다.

**1. `mc` · `mc_seed` · `recent_margin` 열**

`module.mc_shards.combine()` 이 캐시된 샤드를 합쳐 붙입니다. 공정가 시대의 정본 MC 인데,
지금 실제로 쓰이는 것은 `mc_seed`(공통난수)와 열의 존재뿐입니다. `module.mcvar` 가
`D.MC` 를 변형 B(`mc_B`)로 갈아끼우기 때문입니다.

샤드 캐시(`data/cache/mccal_shard_*.json` 28개)가 없으면 재시뮬이 필요하고, 단일코어로 수 시간 걸립니다.

**2. `data/ml.csv` · `data/deeponet.csv`**

`module.data.build_datasets()` 가 parquet 에서 뽑아냅니다. `module.data.load()` 가 읽는 것은
parquet 이 아니라 **이 두 CSV** 입니다.

둘 다 이미 있으면 건너뜁니다.
"""))

cells.append(code("""
from module import data as MD
from module import mc_shards as MS

need_mc = "mc" not in df.columns or df["mc"].isna().all()
nsh = len(list(fm.CACHE.glob("mccal_shard_*.json")))
if need_mc and nsh:
    print("mc 열 없음 | 샤드 캐시 {}개 발견".format(nsh), flush=True)
    df = MS.combine()                                   # 샤드 수 자동감지, 수 초
    print("  mc·mc_seed·recent_margin 기록 완료")
elif need_mc:
    # 샤드 캐시가 없는 원천 전용 트리. mc 는 공정가 시대의 정본 MC 라서 지금 쓰이는 것은
    # mc_seed(build_source 가 부여)뿐이고, D.MC 는 module.mcvar 가 변형 B 로 갈아끼운다.
    # 공정가 2단계 경로만 못 쓰며, module.infer 가 그 경우 명시적으로 거부한다.
    print("mc 열 없음 | 샤드 캐시 0개 — 건너뛴다 (공정가 2단계 경로는 이 트리에서 쓸 수 없다)")
else:
    print("건너뜀  mc 열 (이미 있음)")

p_ml, p_dn = fm.dataset("ml"), fm.dataset("deeponet")
if not (p_ml.exists() and p_dn.exists()):
    print("모델링 데이터셋 생성 중 ...", flush=True)
    info = MD.build_datasets()
    print("  ml {:,}행 × {}열 | deeponet {:,}행 × {}열".format(
        info["ml"], len(info["columns"]["ml"]), info["deeponet"], len(info["columns"]["deeponet"])))
else:
    print("건너뜀  ml.csv · deeponet.csv (이미 있음)")

for p in (fm.DATA / "els3_dataset.parquet", p_ml, p_dn):
    print("  {:1} {:28} {:>8} KB".format(
        "o" if p.exists() else "X", p.name, int(p.stat().st_size / 1e3) if p.exists() else 0))
"""))

cells.append(md("""
## 0-3. 분포 — 계약조건과 시장상태

학습 전에 **입력이 어디에 몰려 있는지** 봅니다. 표본이 거의 없는 구간에서는 모형이 사실상
외삽을 하게 되므로, 나중에 그 구간의 오차가 크더라도 놀랄 일이 아닙니다.
"""))

cells.append(code("""
fig, ax = plt.subplots(2, 4, figsize=(13.2, 6.0))
items = [("fair", "fair value ratio", None), ("tenor", "tenor [years]", None),
         ("coupon", "coupon (annual)", None), ("B", "KI barrier", None),
         ("sig_mean", "mean volatility", None), ("rho", "mean correlation", None),
         ("stepdown", "step-down width", None), ("nobs", "observations", None)]
for a, (c, lab, _) in zip(ax.ravel(), items):
    a.hist(df[c].dropna(), bins=50, color=C1, alpha=.85)
    a.set_xlabel(lab); a.set_ylabel("products")
    a.set_title(lab, fontweight="bold", fontsize=11)
fig.suptitle("Universe distributions - contract terms and market state",
             fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(fm.image("data_distributions"), dpi=200, bbox_inches="tight")
plt.show()
print(df[[c for c, _, _ in items]].describe().round(3).to_string())
"""))

cells.append(md("""
## 0-4. 시간 구성 — 발행 추이와 walk-forward 경계

ELS 발행은 시기별로 쏠립니다. 성능을 무작위 분할로 재면 미래 정보가 새어 들어가므로,
**발행일 순으로 4개 구간을 끊어** 과거로 학습하고 미래로 시험합니다(확장 윈도우).
아래 세로선이 그 경계입니다.
"""))

cells.append(code("""
dt = pd.to_datetime([pd.Timestamp.fromordinal(int(o)) for o in df.isu_ord])
cnt = pd.Series(1, index=dt).resample("QE").sum()
fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.0))
ax[0].bar(cnt.index, cnt.values, width=70, color=C1, edgecolor="white", lw=.5)
ax[0].set_xlabel("issue quarter"); ax[0].set_ylabel("products")
ax[0].set_title("Issuance over time", fontweight="bold")
q = [0.6, 0.7, 0.8, 0.9]
for frac in q:
    ax[0].axvline(dt.sort_values()[int(len(dt) * frac)], color=C2, lw=1.6, ls="--")
ax[0].text(.02, .92, "dashed = walk-forward fold edges", transform=ax[0].transAxes,
           fontsize=9.5, color=C2)

s = pd.Series(df.fair.values, index=dt).resample("QE").mean()
ax[1].plot(s.index, s.values, "-o", color=C1, ms=3.5)
ax[1].axhline(1.0, color="#444", lw=1.2, ls="--")
ax[1].set_xlabel("issue quarter"); ax[1].set_ylabel("mean fair value ratio")
ax[1].set_title("Fair value ratio over time", fontweight="bold")
fig.suptitle("Temporal structure - folds are expanding windows, test is always the future",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .91])
fig.savefig(fm.image("data_time"), dpi=200, bbox_inches="tight")
plt.show()
print("발행기간 {} ~ {}".format(dt.min().date(), dt.max().date()))
"""))

cells.append(md("""
## 0-5. 1·2스타 확장

기초자산이 1~2개인 상품을 **슬롯 중복 패딩**(위 용어 참고)으로 3자산 입력 규약에 수용합니다.

아래 셀이 `generate/build_12star.main()` 으로 직접 만듭니다(이미 있으면 건너뜁니다).
MC 라벨(`mc_variants12`)은 노트북 1 의 1-0 이 붙이고, 번들 구성은 `module.star.make_all()` 이 맡습니다.
"""))

cells.append(code("""
from generate import build_12star

p12 = fm.DATA / "els12_dataset.parquet"
if not p12.exists():
    print("els12_dataset.parquet 생성 중 (약 2분) ...", flush=True)
    build_12star.main()

if p12.exists():
    d12 = pd.read_parquet(p12)
    print("1·2스타 {:,}건 | 기초자산 수별 {}".format(
        len(d12), d12.nstar.value_counts().sort_index().to_dict()))
    lz = d12[[f"lz_barr_{j}" for j in range(12)]].lt(LZ_BARR_NEUTRAL).any(axis=1)
    print("  리자드 {:,}건 ({:.1f}%)".format(int(lz.sum()), 100 * lz.mean()))
    print("  → 1·2스타 리자드 표본이 얇다. 해당 구간 기울기는 신뢰구간을 함께 볼 것.")
else:
    print("els12_dataset.parquet 생성 실패 — 원천·캐시를 확인한다")
"""))

cells.append(md("""
## 결론

1. **유니버스는 4구조 × 기초자산 3개 × KRW** 로 좁힙니다. 공정가비율·만기 범위는 이상치 제거용입니다.
2. **스케줄은 48칸 고정폭**으로 정규화합니다(행사가·지급률·리자드배리어·리자드지급률 각 12칸).
   리자드가 없으면 중립값(배리어 1.2 = 절대 발동 안 함, 지급률 0)으로 채웁니다.
3. **1·2스타는 슬롯 중복 패딩**으로 같은 입력 규약에 들어갑니다. 통합 학습이 왜 필수인지는 2장에서 보입니다.
4. 이론가(MC) 라벨은 1장이 만듭니다. 이 노트북은 **계약과 시장상태까지만** 책임집니다.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "els-abacus", "language": "python",
                                  "name": "els-abacus"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT.name, "| cells", len(cells))
