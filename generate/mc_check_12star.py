# -*- coding: utf-8 -*-
"""스타별 계약조건 섭동 MC 참값 — 1·2·3스타 기준상품에 같은 격자를 적용해 MC 를 다시 돌린다.

기존 contract_sens_mc_check_mono.csv 는 3스타 12개 상품만 담고 있다.
1·2스타 기울기를 재려면 같은 정의의 MC 참값이 스타별로 필요하다.
시장입력(EWMA120 · 영업일120 상관(중복쌍 1.0) · 부트스트랩 커브 · 배당)은
generate/build_mc_variants.py 의 star12 모드와 동일하고, 계약만 CS.GRID 격자로 바꾼다.

산출: result/statistics/mc_check_allstar.csv  (item · nstar · factor · level · mc)
실행: python generate/mc_check_12star.py [스타당 상품수]
"""
import os, sys, time
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module import contract as CS
from module import curve as CV
from module import dividend as DV
from module import features as F
from module import market as MK
from module.seeds import mc_seed_for
from module import mc_engine as ME
from module.data import CONTRACT
from module.schedule import LZ_BARR_NEUTRAL, LZ_PMT_NEUTRAL
from util import file_manager as fm

NPATH = 100_000  # 섭동 격자는 정본 mc_B 와 같은 경로 수
DIV = os.environ.get("MCVAR_DIV", DV.DEFAULT_SRC)  # 라벨과 같은 배당 소스여야 한다
RET = MK.returns()
KBD = MK.calendar(RET)

REN = {"BARR_1/100": "B", "ANL_RTRN/100": "coupon", "TENOR": "tenor"}


def exact_dup_corr(C, ts):
    """중복 슬롯 쌍의 상관을 정확히 1.0 으로 복원 (corr_bd 가 0.999 로 자른 것)."""
    C = np.array(C, float, copy=True)
    for a in range(3):
        for b in range(3):
            if a != b and ts[a] == ts[b]:
                C[a, b] = 1.0
    np.fill_diagonal(C, 1.0)
    return C


def con_of(row):
    """원천 행 -> CONTRACT(51) 벡터 (리자드 결측은 중립값)."""
    v = []
    for c in CONTRACT:
        x = row.get(REN.get(c, c), np.nan)
        if pd.isna(x):
            x = (
                LZ_BARR_NEUTRAL
                if c.startswith("lz_barr_")
                else (LZ_PMT_NEUTRAL if c.startswith("lz_pmt_") else np.nan)
            )
        v.append(float(x))
    return np.array([v], dtype="float32")


def market_of(row):
    """그 상품의 MC 시장입력. 실패하면 None."""
    dt = pd.Timestamp(date.fromordinal(int(row["isu_ord"])))
    ts = [row["udl1"], row["udl2"], row["udl3"]]
    rets = [RET.get(t) for t in ts]
    if any(r is None for r in rets):
        return None
    sig = [F.vol_ewma_lam(x, dt) for x in rets]
    C, _ = F.corr_bd(rets, dt, KBD)
    mi = CV.market_inputs(dt)
    if C is None or np.isnan(sig).any() or mi is None:
        return None
    dz = CV.curve_dz(*CV.bootstrap(mi["cd"], mi["swaps"]))
    got = DV.div_kwargs(ts, dt, float(row["tenor"]), DIV)  # lseg -> 연속 q, yf -> 이산 divlog
    if got is None:
        return None
    return dict(sig=sig, corr=exact_dup_corr(C, ts), dz=dz, dkw=got[0])


def price_con(CONv, mk, row, nobs):
    """CONTRACT 벡터 하나를 MC 로 가격. 스케줄은 CON 에서 되읽는다."""
    k = int(nobs)
    strikes = [float(CONv[0, CONTRACT.index("strk_{}".format(j))]) for j in range(k)]
    pmts = [float(CONv[0, CONTRACT.index("pmt_{}".format(j))]) for j in range(k)]
    lzb = [float(CONv[0, CONTRACT.index("lz_barr_{}".format(j))]) for j in range(k)]
    lzp = [float(CONv[0, CONTRACT.index("lz_pmt_{}".format(j))]) for j in range(k)]
    lzb = [np.nan if b >= LZ_BARR_NEUTRAL else b for b in lzb]
    B = float(CONv[0, CONTRACT.index("BARR_1/100")])
    c = float(CONv[0, CONTRACT.index("ANL_RTRN/100")])
    ten = float(CONv[0, CONTRACT.index("TENOR")])
    return ME.mc_daily_t(
        mk["sig"],
        mk["corr"],
        mk["dz"],
        B,
        strikes,
        ten,
        n=NPATH,
        seed=int(row["mc_seed"]),
        c=c,
        pmts=pmts,
        lz_barr=lzb,
        lz_pmt=lzp,
        **mk["dkw"],
    )


def main(nper=12):
    """스타별 섭동 격자 MC 참값을 만들어 result/statistics/mc_check_allstar.csv 에 저장한다.

    nper: 스타당 기준상품 수 (리자드/스텝 절반씩)."""
    NPER = nper
    d12 = (
        pd.read_parquet(fm.DATA / "els12_dataset.parquet")
        .sort_values("isu_ord")
        .reset_index(drop=True)
    )
    d12["mc_seed"] = mc_seed_for(d12["item"])  # 상품 내용에서 유도 (행 번호 아님)
    d3 = MK.dataset().copy()
    d3["nstar"] = 3
    rng = np.random.default_rng(0)
    picks = []
    for src, ns in [(d12[d12.nstar == 1], 1), (d12[d12.nstar == 2], 2), (d3, 3)]:
        has_lz = src[["lz_barr_{}".format(j) for j in range(12)]].lt(LZ_BARR_NEUTRAL).any(axis=1)
        for want_lz in (True, False):  # 리자드/스텝 절반씩
            pool = src[has_lz] if want_lz else src[~has_lz]
            if not len(pool):
                continue
            take = min(NPER // 2, len(pool))
            picks.append(pool.iloc[rng.choice(len(pool), take, replace=False)].assign(nstar=ns))
    P = pd.concat(picks, ignore_index=True)
    print("기준상품 {} (스타별 {})".format(len(P), P.groupby("nstar").size().to_dict()), flush=True)

    rows, t0 = [], time.time()
    for _, row in P.iterrows():
        mk = market_of(row)
        if mk is None:
            continue
        CON0 = con_of(row)
        nobs = int(row["nobs"])
        has_lz = bool((CON0[0, 24:36] < LZ_BARR_NEUTRAL).any())
        for f, (lab, levels) in CS.GRID.items():
            if f.startswith("lz_") and not has_lz:
                continue
            for lv in levels:
                X = CS.apply_factor(CON0, f, lv, np.array([nobs]), np.array([has_lz]))
                rows.append(
                    dict(
                        item=row["item"],
                        nstar=int(row["nstar"]),
                        factor=f,
                        level=lv,
                        mc=price_con(X, mk, row, nobs) * 10000,
                    )
                )
        print(
            "  {} ({}스타) 누적 {} ({:.0f}s)".format(
                row["item"], int(row["nstar"]), len(rows), time.time() - t0
            ),
            flush=True,
        )
    R = pd.DataFrame(rows)
    R.to_csv(fm.stat("mc_check_allstar"), index=False, encoding="utf-8-sig")
    print(
        "저장 {:,}행 | 스타별 상품수 {}".format(len(R), R.groupby("nstar").item.nunique().to_dict())
    )


if __name__ == "__main__":
    main()
