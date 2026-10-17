# -*- coding: utf-8 -*-
"""0_data_preprocessing.ipynb 생성 — 원천 → 학습 데이터셋.

 흡수: 0_data.ipynb + EDA_raw_filter.ipynb
 실행: python tools/build_nb_data.py
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

DART 원천 공시 → 학습 데이터셋. `module.build_source` 를 불러와 이 노트북 안에서 직접 빌드한다.

| 단계 | 산출 |
|---|---|
| 원천 CSV 3종 + 가격·커브 캐시 | `data/els3_dataset.parquet` (기초자산 3개) |
| 1·2스타 확장 (슬롯 중복 패딩) | `data/els12_dataset.parquet` |
| MC 이론가 라벨 | `data/cache/mcvar/mc_variants.parquet` (1장 참고) |

**유니버스 정의**: 기초자산 3개 · KRW · 4구조(STEP/LIZARD × 낙인/노낙인) ·
공정가비율 0.70~1.05 · 만기 0.5~5.0년.

**노낙인은 `B = 1.0` 으로 인코딩**한다 — 만기 미상환이면 항상 worst 를 수취한다는 뜻이고,
MC 엔진이 `B >= 1.0` 을 명시적으로 '항상 낙인' 으로 처리한다.
"""))

cells.append(code("""
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
REBUILD = False        # True 면 원천에서 다시 빌드 (수 분). False 면 저장본을 읽는다
print("원천:", fm.RAW)
"""))

cells.append(md("""
## 0-1. 필터 퍼널

원천에서 유니버스까지 몇 단계를 거치며 무엇이 왜 떨어지는지 센다.
`build_source` 는 탈락 사유를 `df.attrs["drops"]` 로 돌려준다.
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
## 0-2. 데이터셋 빌드

`build_source()` 가 후보를 돌며 시장입력(변동성·상관·커브)을 붙이고 스케줄을 고정폭 48칸으로
정규화한다. `save=False` 로 돌리면 기존 parquet 를 덮어쓰지 않는다.
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

cells.append(md("## 0-3. 분포 — 계약조건과 시장상태"))

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

cells.append(md("## 0-4. 시간 구성 — 발행 추이와 walk-forward 경계"))

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

기초자산이 1~2개인 상품을 **슬롯 중복 패딩**으로 3자산 입력 규약에 수용한다.
1스타는 A,A,A / 2스타는 A,B,B 로 채우고, 중복쌍의 상관은 정확히 1.0 으로 둔다.
생성은 `tools/build_12star.py`, 번들 구성은 `module.star.make_all()` 이 한다.
"""))

cells.append(code("""
p12 = fm.DATA / "els12_dataset.parquet"
if p12.exists():
    d12 = pd.read_parquet(p12)
    print("1·2스타 {:,}건 | 기초자산 수별 {}".format(
        len(d12), d12.nstar.value_counts().sort_index().to_dict()))
    lz = d12[[f"lz_barr_{j}" for j in range(12)]].lt(LZ_BARR_NEUTRAL).any(axis=1)
    print("  리자드 {:,}건 ({:.1f}%)".format(int(lz.sum()), 100 * lz.mean()))
    print("  → 1·2스타 리자드 표본이 얇다. 해당 구간 기울기는 신뢰구간을 함께 볼 것.")
else:
    print("els12_dataset.parquet 없음 — tools/build_12star.py 로 생성")
"""))

cells.append(md("""
## 결론

1. **유니버스는 4구조 × 기초자산 3개 × KRW** 로 좁힌다. 공정가비율·만기 범위는 이상치 제거용이다.
2. **스케줄은 48칸 고정폭**으로 정규화한다 (행사가·지급률·리자드배리어·리자드지급률 각 12칸).
   리자드가 없으면 중립값(배리어 1.2 = 절대 발동 안 함, 지급률 0)으로 채운다.
3. **1·2스타는 슬롯 중복 패딩**으로 같은 입력 규약에 들어간다 — 2장에서 통합 학습이 필수임을 보인다.
4. 이론가(MC) 라벨은 1장이 만든다. 이 노트북은 **계약과 시장상태까지만** 책임진다.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "els-abacus", "language": "python",
                                  "name": "els-abacus"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT.name, "| cells", len(cells))
