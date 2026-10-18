# -*- coding: utf-8 -*-
"""원천 전용 재현의 판정 — 지수전용 부분집합으로 MC 라벨과 모델 성능을 구본과 대조한다.

**대조 기준선은 `mc_variants_lseg.parquet` 이다.** 구본 스냅샷에는 두 라벨이 있다 —
`mc_variants.parquet`(yf 이산배당)과 `mc_variants_lseg.parquet`(LSEG 연속 q). 신본은 연속 q 이므로
연속 q 기준선과 맞춰야 **가격 계열 교체**와 **경로 수**만 남는다. yf 기준선과 대면 배당 소스 교체
28.26원이 섞여 들어가 판정이 흐려진다.

**왜 지수만 판정하는가.** 가격 계열이 `px_*`(배당 역조정 = 총수익)에서 `Close_Adj`(액면분할만)로
바뀌었다. 개별주식은 두 계열이 실제로 다르므로 라벨이 달라지는 것이 정상이고, 지수는 거의 같다
(^KS200 수익률 상관 0.99994). 그래서 지수전용 상품에서만 재현을 요구할 수 있다.

세 행은 전체를 **분할**한다 (합이 데이터셋 전체 건수).
  TIGHT  세 기초자산 모두 TIGHT        수익률 상관 >= 0.9999 인 6종
  IDX    세 기초자산 모두 IDX, TIGHT 아님   ^HSI · ^STOXX50E 가 섞인 것
  OTHER  개별주식이 하나라도 있는 것        기준 없음, 변화량만 보고

**라벨 차이(mc_mad)로는 재현을 판정할 수 없다.** `mc_seed` 가 isu_ord 정렬 후의 **행 번호**라서
123건이 새로 들어오면서 공통 상품 98.2% 의 난수 추출이 바뀌었다(동일 1,085/58,790). 즉 구본과
신본은 공통난수가 아니고, 라벨 차이는 독립 MC 추정 두 개의 차이 — 입력이 완전히 같아도
평균|차이| 가 EXP_MAD 만큼 남는다. 실측으로도 경로 수를 40k 로 맞춰봤자 줄지 않았다
(TIGHT 9.80원, 오히려 늘었다 — 신본 쪽 40k 잡음이 100k 보다 크다).

**판정은 px_mad 로 한다.** Task 1 의 `source_switch_impact.csv` 는 같은 시드·같은 경로 수로
가격 계열만 바꿔 가격을 매긴 것이어서 MC 잡음이 공통난수로 상쇄된다. 거기서 가격계열 단독 효과는
TIGHT 0.10원(최대 0.49) · IDX 8.15원 · OTHER 24.59원 이다.

**모델 성능은 부분집합별이 아니라 전체(ALL) 기준이다.** 구본 스냅샷에 상품별 예측이 남아 있지
않아(predictions 0개) 부분집합으로 다시 쪼갤 수 없다. 그래서 r2/mape 열은 모든 행에 같은 값이고,
`nb_training_level.csv` 의 group=ALL · model=통합 학습 행에서 가져온다.

산출: result/statistics/raw_run_verification.csv
실행: python generate/verify_raw_run.py
"""
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from util import file_manager as fm  # noqa: E402

BASE = ROOT / "scratch" / "raw_run_baseline"
TIGHT = frozenset({"^N225", "^GSPC", "^NDX", "^HSCE", "^GDAXI", "^KS200"})
IDX = TIGHT | {"^HSI", "^STOXX50E"}
OLD_LABEL = "mc_variants_lseg.parquet"  # 연속 q 기준선 — 배당 소스를 고정한다
LEVEL = "nb_training_level.csv"
ALL_MODEL = "통합 학습 (1·2·3)"
FACE = 10000  # mc_B 는 액면 1 기준 — 원 단위로 환산
NOISE40, NOISE100 = 6.20, 3.62  # 40만 경로를 준참값으로 한 평균|차이| (원), Task 5 실측
EXP_MAD = float(np.hypot(NOISE40, NOISE100))  # 독립 두 추정의 기대 평균|차이| = 7.18원
IMPACT = "source_switch_impact.csv"  # Task 1 의 공통난수 측정 (구본 스냅샷)
# 그 측정이 돌아간 설정. 정본 라벨(10만 경로·연속 q)과 다르므로 판정 출력에 함께 적는다 —
# px_mad 는 "그 배당 처리·경로 수 조건에서의" 가격계열 효과이고 상호작용은 미측정이다.
PX_MEASURED = "40k·이산배당 기준"
METRICS = ("R2", "MAPE", "MAE")  # 모델 성능 재현을 재는 지표
COLS = [
    "subset",
    "n",
    "n_cmp",
    "n_only_new",
    "n_only_old",
    "mc_corr",
    "mc_mad",
    "mc_mad_se",
    "mc_mad_exp",
    "mc_bias",
    "px_n",
    "px_mad",
    "px_se",
    "px_hi",
    "px_bias",
    "r2_old",
    "r2_new",
    "mape_old",
    "mape_new",
    "frozen",
]
FIXTURE = ROOT / "tests" / "fixtures" / "raw_run_verdict.json"  # 기준선 삭제 후의 유일한 근거
BOOT = 2000  # px_mad 신뢰구간용 부트스트랩 반복 (분포가 우측으로 길다)


def _subset_of(row: pd.Series) -> str:
    """상품 하나를 TIGHT / IDX / OTHER 중 하나로."""
    tks = {row["udl1"], row["udl2"], row["udl3"]}
    if tks <= TIGHT:
        return "TIGHT"
    return "IDX" if tks <= IDX else "OTHER"


def _level(path: str | Path) -> pd.DataFrame | None:
    """nb_training_level.csv 의 ALL · 통합학습 행에서 (R2, MAPE). 없으면 (nan, nan)."""
    if not Path(path).exists():
        return np.nan, np.nan
    L = pd.read_csv(path)
    r = L[(L.group == "ALL") & (L.model == ALL_MODEL)]
    if not len(r):
        return np.nan, np.nan
    return float(r.R2.iloc[0]), float(r.MAPE.iloc[0])


def px_effect() -> dict[str, tuple[int, float, float, float, float]]:
    """부분집합 -> (n, 평균|차이|, 편향). 공통난수로 MC 잡음이 상쇄된 가격계열 단독 효과.

    Task 1 이 같은 시드·같은 경로 수로 `mc_base`(구 가격계열)와 `mc_px`(신 가격계열)를
    나란히 매긴 결과를 읽는다. 파일이 없으면 전부 nan."""
    p = BASE / "statistics" / IMPACT
    if not p.exists():
        d = _fixture()
        if d is None:
            return {}
        return {
            s: (
                g["n"],
                g["mad"],
                g["bias"],
                np.nan if g["se"] is None else g["se"],
                np.nan if g["hi"] is None else g["hi"],
            )
            for s, g in d["px_effect"].items()
        }
    S = pd.read_csv(p)
    d = pd.read_parquet(
        fm.DATA / "els3_dataset.parquet", columns=["item", "udl1", "udl2", "udl3"]
    ).set_index("item")
    S = S.join(d, on="item")
    S["subset"] = [_subset_of(r) for _, r in S.iterrows()]
    e = S.mc_px - S.mc_base
    out = {}
    rng = np.random.default_rng(0)
    for sub_, g in S.groupby("subset"):
        a = e[g.index].abs().to_numpy(float)
        hi = np.nan
        if len(a) > 2:
            # 분포가 우측으로 길다(지수전용에서도 최대 135원). 정규 SE 는 구간을 과소추정하므로
            # 평균의 백분위 부트스트랩 상한(95% 단측)을 함께 낸다.
            bs = a[rng.integers(0, len(a), size=(BOOT, len(a)))].mean(axis=1)
            hi = float(np.percentile(bs, 95))
        out[sub_] = (
            len(a),
            float(a.mean()),
            float(e[g.index].mean()),
            float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan,
            hi,
        )
    return out


def _fixture() -> dict[str, Any] | None:
    """고정된 판정 근거. 없으면 None."""
    import json

    if not FIXTURE.exists():
        return None
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frozen_table() -> pd.DataFrame | None:
    """기준선 스냅샷이 사라진 뒤의 대조표 — generate/freeze_verdict.py 가 고정한 값에서 만든다.

    재계산이 아니라 **기록된 판정 근거**다. frozen 열로 그 사실을 표에 남긴다."""
    d = _fixture()
    if d is None:
        raise FileNotFoundError(
            "기준선 스냅샷({})도 고정 근거({})도 없다 — 판정을 재구성할 수 없다".format(
                BASE, FIXTURE
            )
        )
    mo = d["model_old"]
    r2_n, mp_n = _level(fm.stat(LEVEL.replace(".csv", "")))
    rows = []
    for sub_ in ("TIGHT", "IDX", "OTHER"):
        px, lb = d["px_effect"][sub_], d["label_summary"][sub_]
        rows.append(
            dict(
                # n(부분집합 전체 건수)은 고정 근거에 담지 않았다. 센티넬(-1) 대신 비워 둔다 —
                # 사람이 읽는 표에 -1 이 섞이면 수치로 오독된다.
                subset=sub_,
                n=np.nan,
                n_cmp=lb["n_cmp"],
                n_only_new=lb["n_only_new"],
                n_only_old=lb["n_only_old"],
                mc_corr=lb["mc_corr"],
                mc_mad=lb["mc_mad"],
                mc_mad_se=lb["mc_mad_se"],
                mc_mad_exp=EXP_MAD,
                mc_bias=lb["mc_bias"],
                px_n=px["n"],
                px_mad=px["mad"],
                px_se=np.nan if px["se"] is None else px["se"],
                px_hi=np.nan if px["hi"] is None else px["hi"],
                px_bias=px["bias"],
                r2_old=mo["R2"]["value"],
                r2_new=r2_n,
                mape_old=mo["MAPE"]["value"],
                mape_new=mp_n,
                frozen=True,
            )
        )
    return pd.DataFrame(rows)[COLS]


def compare() -> pd.DataFrame:
    """세 부분집합의 MC 라벨 대조표. 열은 COLS 로 고정."""
    if not (BASE / "cache" / "mcvar" / OLD_LABEL).exists():
        return _frozen_table()  # 스냅샷이 삭제됐다 — 고정 근거로 판정한다
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet", columns=["item", "udl1", "udl2", "udl3"])
    d["subset"] = d.apply(_subset_of, axis=1)

    # **구본 데이터셋도 합집합에 넣는다.** 신본 기준으로 조인하면 신본에서 사라진 상품을
    # 구조적으로 볼 수 없다 — 이 브랜치가 막으려는 위험이 바로 조용한 상품 손실이다.
    od = pd.read_parquet(
        BASE / "datasets" / "els3_dataset.parquet", columns=["item", "udl1", "udl2", "udl3"]
    )
    d["in_new"] = True
    lost = od[~od.item.isin(set(d.item))].copy()
    if len(lost):
        lost["subset"] = lost.apply(_subset_of, axis=1)
        lost["in_new"] = False
        d = pd.concat([d, lost], ignore_index=True)
    d["in_old"] = d.item.isin(set(od.item))

    new = (
        pd.read_parquet(fm.CACHE / "mcvar" / "mc_variants.parquet", columns=["item", "mc_B"])
        .set_index("item")
        .mc_B
    )
    old = (
        pd.read_parquet(BASE / "cache" / "mcvar" / OLD_LABEL, columns=["item", "mc_B"])
        .set_index("item")
        .mc_B
    )

    J = d.join(new.rename("new"), on="item").join(old.rename("old"), on="item")
    r2_o, mp_o = _level(BASE / "statistics" / LEVEL)
    r2_n, mp_n = _level(fm.stat(LEVEL.replace(".csv", "")))

    PX = px_effect()
    rows = []
    for s in ("TIGHT", "IDX", "OTHER"):
        g = J[J.subset == s]
        both = g.dropna(subset=["new", "old"])
        dif = (both.new - both.old) * FACE
        pn, pm, pb, pse, phi = PX.get(s, (0, np.nan, np.nan, np.nan, np.nan))
        ad = dif.abs()
        rows.append(
            dict(
                subset=s,
                n=int(g.in_new.sum()),
                n_cmp=len(both),
                # 신규/손실은 **데이터셋 소속**으로 센다. 라벨 결측과 섞으면 수가 맞지 않는다
                # (종전에는 구본 라벨이 NaN 이던 18행까지 더해 141 로 나왔다. 실제 유입은 123).
                n_only_new=int((g.in_new & ~g.in_old).sum()),
                n_only_old=int((~g.in_new & g.in_old).sum()),
                mc_corr=float(both.new.corr(both.old)) if len(both) > 2 else np.nan,
                mc_mad=float(ad.mean()) if len(both) else np.nan,
                mc_mad_se=float(ad.std(ddof=1) / np.sqrt(len(ad))) if len(ad) > 1 else np.nan,
                mc_mad_exp=EXP_MAD,
                mc_bias=float(dif.mean()) if len(both) else np.nan,
                px_n=pn,
                px_mad=pm,
                px_se=pse,
                px_hi=phi,
                px_bias=pb,
                r2_old=r2_o,
                r2_new=r2_n,
                mape_old=mp_o,
                mape_new=mp_n,
                frozen=False,
            )
        )
    return pd.DataFrame(rows)[COLS]


GATE = {"TIGHT": 5.0, "IDX": 10.0}  # px_mad 상한 (원). 계획서의 판정선을 그대로 쓴다.


def _residual_note(r: pd.Series) -> str:
    """라벨차와 잡음기대의 차이를 설명하거나 미설명분으로 밝힌다.

    EXP_MAD 의 두 입력은 40만 경로 **준참값 대비** 측정치라 기준 자체의 분산을 이미 품고 있다.
    따라서 EXP_MAD 는 기대 격차를 과대추정하고, 아래 미설명분은 **하한**이다.
    또 EXP_MAD 는 스칼라 하나인데 MC 잡음은 가격 분산에 비례하므로 OTHER 에서는 더 크다 —
    부분집합별로 다시 재지 않았으므로 TIGHT 밖에서는 참고치로만 본다."""
    resid = r.mc_mad - r.mc_mad_exp
    if not np.isfinite(r.mc_mad_se) or r.mc_mad_se <= 0:
        return "라벨차 {:.2f} vs 잡음기대 {:.2f}".format(r.mc_mad, r.mc_mad_exp)
    z = resid / r.mc_mad_se
    if abs(z) <= 3:
        return "라벨차 {:.2f}+-{:.2f} ~ 잡음기대 {:.2f} ({:.1f} SE)".format(
            r.mc_mad, r.mc_mad_se, r.mc_mad_exp, abs(z)
        )
    return "라벨차 {:.2f}+-{:.2f} vs 잡음기대 {:.2f} -> {:+.2f}원 미설명 ({:.0f} SE, 하한)".format(
        r.mc_mad, r.mc_mad_se, r.mc_mad_exp, resid, abs(z)
    )


def verdict(T: pd.DataFrame) -> bool | None:
    """부분집합별 (판정, 설명). True 통과 · None 미결론/기준없음 · False 미달.

    판정량은 px_mad — 공통난수로 MC 잡음이 상쇄된 가격계열 단독 효과다.
    mc_mad 는 구본·신본이 공통난수가 아니라 판정에 쓸 수 없고, 설명에 참고로만 붙인다.

    **점추정이 아니라 구간으로 판정한다.** px_mad 는 200건 표본(TIGHT 11 · IDX 89)의
    평균이므로 표본오차가 있다. 95% 단측 부트스트랩 상한(px_hi)이 기준을 넘으면 '통과' 가
    아니라 **미결론**이다 — 점추정만 보면 동전던지기를 통과로 적게 된다.

    **기준선의 유래에 주의.** 5/10원은 계획서가 mc_mad(잡음 포함)에 대해 "10만 경로 잡음
    3.62원의 약 1.4배" 로 잡은 값이다. px_mad 는 공통난수로 잡음이 상쇄된 **계통 편차**이므로
    그 근거가 그대로 옮겨오지 않는다. TIGHT(0.10원, 기준의 1/50)는 어떤 기준을 써도 통과지만,
    IDX(8.15원)는 기준 자체를 다시 유도해야 결론을 낼 수 있다."""
    out = {}
    for _, r in T.iterrows():
        note = _residual_note(r)
        if r.subset not in GATE:
            out[r.subset] = (
                None,
                "기준 없음 — 가격계열 {:.2f}원 ({}) | {}".format(r.px_mad, PX_MEASURED, note),
            )
            continue
        m_max = GATE[r.subset]
        if not np.isfinite(r.px_mad):
            out[r.subset] = (False, "가격계열을 재지 못했다 | {}".format(note))
            continue
        hi = r.px_hi if np.isfinite(r.px_hi) else r.px_mad
        if hi <= m_max:
            ok, how = True, "상한 {:.2f} <= {}원".format(hi, m_max)
        elif r.px_mad <= m_max:
            ok, how = None, "점추정 {:.2f} <= {}원 이나 상한 {:.2f} 초과 — **미결론**".format(
                r.px_mad, m_max, hi
            )
        else:
            ok, how = False, "점추정 {:.2f} > {}원".format(r.px_mad, m_max)
        out[r.subset] = (
            ok,
            "가격계열 {:.2f}+-{:.2f} (n={:.0f}, {}) | {} | {}".format(
                r.px_mad, r.px_se, r.px_n, PX_MEASURED, how, note
            ),
        )
    return out


def _all_row(path: str | Path) -> pd.Series | None:
    """nb_training_level.csv 의 ALL · 통합학습 행. 없으면 None."""
    if not Path(path).exists():
        return None
    L = pd.read_csv(path)
    r = L[(L.group == "ALL") & (L.model == ALL_MODEL)]
    return r.iloc[0] if len(r) else None


def model_verdict() -> list[dict[str, Any]]:
    """지표 -> (판정, z, 설명). **2표본 검정**이다.

    구본 성능도 시드 5개의 추정값이므로 참값처럼 쓸 수 없다. 계획서는 구본 SE 만으로
    +-2SE 를 재라고 했는데, 그러면 신본 쪽 시드 분산을 0 으로 가정하는 셈이다.
    실측에서 신본 MAPE_se(0.0378)가 구본(0.0155)의 2.4배라 차이가 크다 —
    한쪽만 쓰면 '밖', 결합하면 0.84 SE 로 '안' 이다. mc_mad 게이트와 같은 종류의 오류다.

    z = (신본 - 구본) / hypot(구본 SE, 신본 SE).

    **|z| <= 2 는 "구별되지 않는다" 이지 "같다" 가 아니다.** 시드 5개짜리 두 표본이라
    검정력이 낮다 — 검출 가능한 최소 차이(2 x 결합 SE)를 설명에 함께 적는다. R2 의 경우
    그 한계가 0.014 인데 실측 하락은 0.0064 로, **관측된 하락의 두 배**가 검출 한계다.
    그러니 "재현" 은 "학습 난수 변동과 구별되지 않는다" 로 읽어야 한다.

    **두 팔은 다른 상품 집합에서 측정됐다.** 구본 n=29,743, 신본 n=29,916 으로 173건이 다르고
    (^SX7E 신규분 포함), 시드 SE 는 그 차이를 전혀 설명하지 못한다. 그러므로 |z| <= 2 는
    재현 검정이 아니라 "서로 다른 두 양이 학습 잡음보다 덜 다르다" 는 진술이다.

    한 가지 더: 시드 SE 는 **학습 난수**만 재고, 라벨 교체는 다섯 시드에 공통인 고정효과다.
    즉 이 검정은 검정 대상에 구조적으로 눈을 감는다. 그래서 하락 **방향**도 함께 보고한다 —
    10만 경로 라벨은 4만보다 덜 시끄러우므로 달성 가능한 R2 는 올라야 하는데 떨어졌다.
    구별 불가라도 열린 항목이다."""
    ro = _all_row(BASE / "statistics" / LEVEL)
    frozen, n_old = False, None
    if ro is None:
        d = _fixture()
        if d is not None:
            mo = d["model_old"]
            ro = pd.Series(
                {k: v for m in METRICS for k, v in ((m, mo[m]["value"]), (m + "_se", mo[m]["se"]))}
            )
            frozen, n_old = True, mo.get("n")  # 왼쪽 팔은 측정이 아니라 **기록**이다
    elif ro is not None:
        n_old = int(ro["n"])
    rn = _all_row(fm.stat(LEVEL.replace(".csv", "")))
    BETTER_IS_UP = {"R2": True, "MAPE": False, "MAE": False}  # 라벨 잡음이 줄면 기대되는 방향
    out = {}
    for m in METRICS:
        if ro is None or rn is None:
            out[m] = (False, np.nan, "성능 표가 없다")
            continue
        se = float(np.hypot(ro[m + "_se"], rn[m + "_se"]))
        d = float(rn[m] - ro[m])
        z = d / se if se > 0 else np.nan
        worse = (d < 0) if BETTER_IS_UP[m] else (d > 0)
        note = " | 방향: 예상과 반대(라벨 잡음이 줄었는데 나빠졌다)" if worse else ""
        # 검출한계는 **80% 검정력** 기준(2.8 SE)으로 적는다. 2 SE 는 검정력 50% 짜리 바닥이라
        # '구별 불가' 를 과독하게 만든다 — 바닥을 적는 목적 자체가 그걸 막는 것이다.
        pop = ""
        if n_old is not None and int(rn["n"]) != n_old:
            pop = " | 모집단 다름: n {:,} -> {:,} ({:+,}건)".format(
                n_old, int(rn["n"]), int(rn["n"]) - n_old
            )
        out[m] = (
            bool(abs(z) <= 2.0),
            z,
            "{}{:.4f}+-{:.4f} -> {:.4f}+-{:.4f} | {:+.4f} = {:.2f} SE "
            "| 검출한계 {:.4f} (80% 검정력){}{}".format(
                "[기록] " if frozen else "",
                ro[m],
                ro[m + "_se"],
                rn[m],
                rn[m + "_se"],
                d,
                abs(z),
                2.8 * se,
                pop,
                note,
            ),
        )
    return out


def main() -> None:
    """재현 판정을 모아 CSV 로 쓰고 사람이 읽을 표를 찍는다.

    입력
      없음 (기준선 스냅샷 또는 고정된 픽스처 + 현재 라벨·성능표)
    출력
      없음 (result/statistics/raw_run_verification.csv 와 표준출력).
            표준출력을 UTF-8 로 재설정한다 — 윈도우 기본 cp949 는 '—'·'±' 를 찍지 못해
            CSV 를 쓴 **뒤에** UnicodeEncodeError 로 죽고, 그러면 인코딩 때문에
            판정이 '실패' 로 끝난다
    """
    # 윈도우 기본 콘솔은 cp949 라 — 나 ± 같은 글자를 찍지 못해 UnicodeEncodeError 로 죽는다.
    # CSV 를 먼저 쓰고 그 뒤에 죽으므로, 이걸 게이트로 걸면 인코딩 때문에 '실패' 로 끝난다.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    T = compare()
    fm.stat(LEVEL.replace(".csv", ""))  # statistics 디렉터리 보장
    out = fm.stat("raw_run_verification")
    T.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(T.to_string(index=False, float_format=lambda x: "{:.5f}".format(x)))
    print()
    gated = set(GATE)
    for s, (ok, why) in verdict(T).items():
        # 세 상태를 세 단어로. 게이트가 걸렸는데 결론을 못 낸 것(미결론)과
        # 애초에 게이트가 없는 것(기준없음)은 다른 말이어야 한다.
        mark = (
            ("통과" if ok else "미달")
            if ok is not None
            else ("미결론" if s in gated else "기준없음")
        )
        print("  {:6} {:8} {}".format(s, mark, why))

    print()
    print("  모델 성능 (ALL · 통합학습, 2표본 검정 |z| <= 2)")
    for m, (ok, z, why) in model_verdict().items():
        print("    {:5} {:4} {}".format(m, "재현" if ok else "변화", why))
    print()
    print("저장", out)
    return T


if __name__ == "__main__":
    main()
