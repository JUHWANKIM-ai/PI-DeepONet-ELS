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
 실행: python tools/verify_raw_run.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from util import file_manager as fm      # noqa: E402

BASE = ROOT / "scratch" / "raw_run_baseline"
TIGHT = frozenset({"^N225", "^GSPC", "^NDX", "^HSCE", "^GDAXI", "^KS200"})
IDX = TIGHT | {"^HSI", "^STOXX50E"}
OLD_LABEL = "mc_variants_lseg.parquet"          # 연속 q 기준선 — 배당 소스를 고정한다
LEVEL = "nb_training_level.csv"
ALL_MODEL = "통합 학습 (1·2·3)"
FACE = 10000                                     # mc_B 는 액면 1 기준 — 원 단위로 환산
NOISE40, NOISE100 = 6.20, 3.62      # 40만 경로를 준참값으로 한 평균|차이| (원), Task 5 실측
EXP_MAD = float(np.hypot(NOISE40, NOISE100))   # 독립 두 추정의 기대 평균|차이| = 7.18원
IMPACT = "source_switch_impact.csv"            # Task 1 의 공통난수 측정 (구본 스냅샷)
METRICS = ("R2", "MAPE", "MAE")                # 모델 성능 재현을 재는 지표
COLS = ["subset", "n", "n_cmp", "n_only_new", "mc_corr", "mc_mad", "mc_mad_exp", "mc_bias",
        "px_n", "px_mad", "px_bias", "r2_old", "r2_new", "mape_old", "mape_new"]


def _subset_of(row):
    """상품 하나를 TIGHT / IDX / OTHER 중 하나로."""
    tks = {row["udl1"], row["udl2"], row["udl3"]}
    if tks <= TIGHT:
        return "TIGHT"
    return "IDX" if tks <= IDX else "OTHER"


def _level(path):
    """nb_training_level.csv 의 ALL · 통합학습 행에서 (R2, MAPE). 없으면 (nan, nan)."""
    if not Path(path).exists():
        return np.nan, np.nan
    L = pd.read_csv(path)
    r = L[(L.group == "ALL") & (L.model == ALL_MODEL)]
    if not len(r):
        return np.nan, np.nan
    return float(r.R2.iloc[0]), float(r.MAPE.iloc[0])


def px_effect():
    """부분집합 -> (n, 평균|차이|, 편향). 공통난수로 MC 잡음이 상쇄된 가격계열 단독 효과.

     Task 1 이 같은 시드·같은 경로 수로 `mc_base`(구 가격계열)와 `mc_px`(신 가격계열)를
     나란히 매긴 결과를 읽는다. 파일이 없으면 전부 nan."""
    p = BASE / "statistics" / IMPACT
    if not p.exists():
        return {}
    S = pd.read_csv(p)
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet",
                        columns=["item", "udl1", "udl2", "udl3"]).set_index("item")
    S = S.join(d, on="item")
    S["subset"] = [_subset_of(r) for _, r in S.iterrows()]
    e = S.mc_px - S.mc_base
    return {s: (len(g), float(e[g.index].abs().mean()), float(e[g.index].mean()))
            for s, g in S.groupby("subset")}


def compare():
    """세 부분집합의 MC 라벨 대조표. 열은 COLS 로 고정."""
    d = pd.read_parquet(fm.DATA / "els3_dataset.parquet",
                        columns=["item", "udl1", "udl2", "udl3"])
    d["subset"] = d.apply(_subset_of, axis=1)

    new = pd.read_parquet(fm.CACHE / "mcvar" / "mc_variants.parquet",
                          columns=["item", "mc_B"]).set_index("item").mc_B
    old = pd.read_parquet(BASE / "cache" / "mcvar" / OLD_LABEL,
                          columns=["item", "mc_B"]).set_index("item").mc_B

    J = d.join(new.rename("new"), on="item").join(old.rename("old"), on="item")
    r2_o, mp_o = _level(BASE / "statistics" / LEVEL)
    r2_n, mp_n = _level(fm.stat(LEVEL.replace(".csv", "")))

    PX = px_effect()
    rows = []
    for s in ("TIGHT", "IDX", "OTHER"):
        g = J[J.subset == s]
        both = g.dropna(subset=["new", "old"])
        dif = (both.new - both.old) * FACE
        pn, pm, pb = PX.get(s, (0, np.nan, np.nan))
        rows.append(dict(
            subset=s, n=len(g), n_cmp=len(both),
            n_only_new=int(g.new.notna().sum() - len(both)),
            mc_corr=float(both.new.corr(both.old)) if len(both) > 2 else np.nan,
            mc_mad=float(dif.abs().mean()) if len(both) else np.nan,
            mc_mad_exp=EXP_MAD,
            mc_bias=float(dif.mean()) if len(both) else np.nan,
            px_n=pn, px_mad=pm, px_bias=pb,
            r2_old=r2_o, r2_new=r2_n, mape_old=mp_o, mape_new=mp_n))
    return pd.DataFrame(rows)[COLS]


GATE = {"TIGHT": 5.0, "IDX": 10.0}      # px_mad 상한 (원). 계획서의 판정선을 그대로 쓴다.


def verdict(T):
    """부분집합별 (판정, 설명). OTHER 는 기준 없음.

     판정량은 px_mad — 공통난수로 MC 잡음이 상쇄된 가격계열 단독 효과다.
     mc_mad 는 구본·신본이 공통난수가 아니라 판정에 쓸 수 없고, 설명에 참고로만 붙인다."""
    out = {}
    for _, r in T.iterrows():
        if r.subset not in GATE:
            out[r.subset] = (None, "기준 없음 — 가격계열 {:.2f}원 (라벨차 {:.2f})".format(
                r.px_mad, r.mc_mad))
            continue
        m_max = GATE[r.subset]
        ok = bool(np.isfinite(r.px_mad) and r.px_mad <= m_max)
        out[r.subset] = (ok, "가격계열 {:.2f} <= {}원 (n={:.0f}) | 라벨차 {:.2f} ~ 잡음기대 {:.2f}".format(
            r.px_mad, m_max, r.px_n, r.mc_mad, r.mc_mad_exp))
    return out


def _all_row(path):
    """nb_training_level.csv 의 ALL · 통합학습 행. 없으면 None."""
    if not Path(path).exists():
        return None
    L = pd.read_csv(path)
    r = L[(L.group == "ALL") & (L.model == ALL_MODEL)]
    return r.iloc[0] if len(r) else None


def model_verdict():
    """지표 -> (판정, z, 설명). **2표본 검정**이다.

     구본 성능도 시드 5개의 추정값이므로 참값처럼 쓸 수 없다. 계획서는 구본 SE 만으로
     +-2SE 를 재라고 했는데, 그러면 신본 쪽 시드 분산을 0 으로 가정하는 셈이다.
     실측에서 신본 MAPE_se(0.0378)가 구본(0.0155)의 2.4배라 차이가 크다 —
     한쪽만 쓰면 '밖', 결합하면 0.84 SE 로 '안' 이다. mc_mad 게이트와 같은 종류의 오류다.

     z = (신본 - 구본) / hypot(구본 SE, 신본 SE). |z| <= 2 면 재현으로 본다."""
    ro = _all_row(BASE / "statistics" / LEVEL)
    rn = _all_row(fm.stat(LEVEL.replace(".csv", "")))
    out = {}
    for m in METRICS:
        if ro is None or rn is None:
            out[m] = (False, np.nan, "성능 표가 없다")
            continue
        se = float(np.hypot(ro[m + "_se"], rn[m + "_se"]))
        z = float((rn[m] - ro[m]) / se) if se > 0 else np.nan
        out[m] = (bool(abs(z) <= 2.0), z, "{:.4f}+-{:.4f} -> {:.4f}+-{:.4f} | {:+.4f} = {:.2f} SE".format(
            ro[m], ro[m + "_se"], rn[m], rn[m + "_se"], rn[m] - ro[m], abs(z)))
    return out


def main():
    T = compare()
    fm.stat(LEVEL.replace(".csv", ""))          # statistics 디렉터리 보장
    out = fm.stat("raw_run_verification")
    T.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(T.to_string(index=False, float_format=lambda x: "{:.5f}".format(x)))
    print()
    for s, (ok, why) in verdict(T).items():
        mark = "기준없음" if ok is None else ("통과" if ok else "미달")
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
