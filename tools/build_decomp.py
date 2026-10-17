# -*- coding: utf-8 -*-
"""ΔMC 요인분해 — 어느 입력이 공정가와의 격차를 줄였나 (노트북 3 범주 3).

 변형 A(NS 커브 · HV120 · q=0) 에서 변형 B(부트스트랩 커브 · EWMA120 · 이산배당) 로 갈 때
 가격 변화를 **입력별로** 떼어낸다. 같은 상품·같은 공통난수에서 한 번에 하나씩만 바꾼다.

   ns_h120_q0     NS 커브 · HV120 · 배당 0          (출발점 = 변형 A)
   ns_h120        + 이산배당                         배당 효과
   bt_disc_h120   + 할인커브만 부트스트랩            커브(할인) 효과   ← mc_engine 의 disc_beta
   bt_h120        + 드리프트도 부트스트랩            커브(드리프트) 효과
   bt_e120        + HV120 -> EWMA120                변동성 효과 (= 변형 B)

 할인과 드리프트를 나누는 것이 요점이다 — 둘은 서로 상쇄돼서 커브 교체의 순효과가 작다.

 종전에는 생성기가 scratch 에만 있어 result/statistics/mc_boot_curve_200.csv 가
 **되읽기만 가능한 고아 파일**로 남아 있었다 (노트북 3 이 이 파일에 의존한다).

 표본: els3_dataset 에서 시장입력이 모두 조립되는 상품을 발행일 순으로 등간격 추출.
 실행:  python tools/build_decomp.py build [n]      새로 산출 (기본 200)
        python tools/build_decomp.py verify         기존 CSV 의 상품을 다시 가격해 대조
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module import curve as CV
from module import dividend as DV
from module import features as F
from module import market as MK
from module import mc_engine as ME
from util import file_manager as fm

NPATH = 40_000              # 정본 mc_A/mc_B 와 같은 경로 수
NAME = "mc_boot_curve_200"
FACE = 10000
COLS = ["item", "isu_dt", "tenor", "fair", "hv120", "ew120", "rho120",
        "ns_zT", "bt_zT", "ns_h120_q0", "ns_h120", "bt_disc_h120", "bt_h120", "bt_e120"]


def decomp_row(r, ret, cal):
    """상품 하나의 5개 가격 + 입력 요약. 시장입력이 하나라도 없으면 None."""
    dt = pd.Timestamp.fromordinal(int(r["isu_ord"]))
    tks = [r["udl1"], r["udl2"], r["udl3"]]
    rets = [ret.get(t) for t in tks]
    if any(x is None for x in rets):
        return None
    h120 = [F.vol_win(x, dt, 120) for x in rets]
    e120 = [F.vol_ewma_lam(x, dt) for x in rets]
    rho, _ = F.corr_bd(rets, dt, cal)
    dz_ns, mi = MK.ns_zero(dt), CV.market_inputs(dt)
    if rho is None or dz_ns is None or mi is None or np.isnan(h120 + e120).any():
        return None
    dz_bt = CV.curve_dz(*CV.bootstrap(mi["cd"], mi["swaps"]))
    strikes, pmts, lzb, lzp = MK.schedule(r)
    ten, B = float(r["tenor"]), float(r["B"])
    divlog, _, _, _ = DV.build_div(tks, dt, ten)
    kw = dict(n=NPATH, seed=int(r["mc_seed"]), c=float(r["coupon"]),
              pmts=pmts, lz_barr=lzb, lz_pmt=lzp)

    def px(sig, beta, divl=None, disc=None):
        return float(ME.mc_daily_t(sig, rho, beta, B, strikes, ten,
                                   divlog=divl, disc_beta=disc, **kw))

    return dict(item=r["item"], isu_dt=dt.date().isoformat(), tenor=ten, fair=float(r["fair"]),
                hv120=float(np.mean(h120)), ew120=float(np.mean(e120)),
                rho120=float(np.mean([rho[a, b] for a, b in zip(*np.triu_indices(3, 1))])),
                ns_zT=float(dz_ns(ten)[0]), bt_zT=float(dz_bt(ten)[0]),
                ns_h120_q0=px(h120, dz_ns),
                ns_h120=px(h120, dz_ns, divl=divlog),
                bt_disc_h120=px(h120, dz_ns, divl=divlog, disc=dz_bt),
                bt_h120=px(h120, dz_bt, divl=divlog),
                bt_e120=px(e120, dz_bt, divl=divlog))


def components(B):
    """노트북 3 과 같은 분해 (원 단위)."""
    return pd.DataFrame({
        "dividends": (B.ns_h120 - B.ns_h120_q0) * FACE,
        "curve: discount": (B.bt_disc_h120 - B.ns_h120) * FACE,
        "curve: drift": (B.bt_h120 - B.bt_disc_h120) * FACE,
        "curve: net": (B.bt_h120 - B.ns_h120) * FACE,
        "HV -> EWMA": (B.bt_e120 - B.bt_h120) * FACE,
        "total": (B.bt_e120 - B.ns_h120_q0) * FACE})


def run(items=None, n=200):
    df = MK.dataset()
    ret = MK.returns()
    cal = MK.calendar(ret)
    if items is not None:
        sel = df.index[df.item.isin(set(items))].values
    else:
        sel = np.arange(len(df))
    rows, t0 = [], time.time()
    for j, i in enumerate(sel):
        rec = decomp_row(df.iloc[int(i)], ret, cal)
        if rec is not None:
            rows.append(rec)
        if items is None and len(rows) >= n:
            break
        if items is None and j and j % 500 == 0:
            print("  훑은 행 {:,} / 모은 표본 {} ({:.0f}s)".format(j, len(rows), time.time() - t0), flush=True)
    return pd.DataFrame(rows)[COLS], time.time() - t0


def build(n=200):
    """등간격 표본으로 새로 산출.

     발행일 순으로 len//n 간격의 자리를 잡고, 그 자리 상품의 시장입력이 없으면 **다음 행으로 밀어서**
     채운다. 그래서 자리 간격이 293·294·296 처럼 들쭉날쭉해진다 — 기존 CSV 와 같은 규칙이다."""
    df = MK.dataset()
    ret = MK.returns()
    cal = MK.calendar(ret)
    step = max(1, len(df) // n)
    rows, t0, i = [], time.time(), 0
    for k in range(n):
        i = max(i, k * step)
        while i < len(df):
            rec = decomp_row(df.iloc[i], ret, cal)
            i += 1
            if rec is not None:
                rows.append(rec)
                break
        if i >= len(df):
            break
    T = pd.DataFrame(rows)[COLS]
    T.to_csv(fm.stat(NAME), index=False, encoding="utf-8-sig")
    print("저장 {} ({}건, {:.1f}분)".format(fm.stat(NAME).name, len(T), (time.time() - t0) / 60))
    print(components(T).describe().T[["mean", "std", "min", "max"]].round(1).to_string())
    return T


def verify():
    """기존 CSV 의 상품을 다시 가격해 열별로 대조 (생성기가 그 파일을 만들 수 있는지)."""
    O = pd.read_csv(fm.stat(NAME), encoding="utf-8-sig")
    T, el = run(items=O.item.values)
    J = O.merge(T, on="item", suffixes=("_old", "_new"))
    print("대조 {}건 / 기존 {}건 ({:.1f}분)".format(len(J), len(O), el / 60))
    for c in ["hv120", "ew120", "ns_zT", "bt_zT", "ns_h120_q0", "ns_h120",
              "bt_disc_h120", "bt_h120", "bt_e120"]:
        a, b = J[c + "_old"].values, J[c + "_new"].values
        print("  {:14} 최대차이 {:.3e} | 평균차이 {:+.3e}".format(c, np.nanmax(np.abs(a - b)), np.nanmean(b - a)))
    print()
    print("분해 평균 (원) — 기존 vs 재산출")
    co, cn = components(O), components(T)
    for k in co.columns:
        print("  {:17} {:+8.1f} -> {:+8.1f}  (|평균| {:7.1f} -> {:7.1f})".format(
            k, co[k].mean(), cn[k].mean(), co[k].abs().mean(), cn[k].abs().mean()))
    return J


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else:
        verify()
