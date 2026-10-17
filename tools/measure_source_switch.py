# -*- coding: utf-8 -*-
"""교체 영향 측정 — 원천 전용 재현이 라벨을 얼마나 흔드는지 요인별로 분리한다.

 원천 전용(data/raw + data/share_underlyings_v2)으로 가려면 세 가지가 함께 바뀐다.

   가격계열  px_*(배당 역조정, 총수익)        -> Close_Adj(분할만 조정)
   배당      이산배당 divlog                  -> 연속 배당수익률 qcont
   경로 수   40,000                           -> 100,000

 셋을 한꺼번에 바꾸면 어느 것이 얼마나 흔들었는지 알 수 없다. 그래서 기준에서
 **하나씩만** 바꾼 가격을 함께 내고, 셋 다 바꾼 것을 덧붙인다.
 커브·계약·공통난수(mc_seed)는 전부 고정하므로 차이는 오직 그 요인에서 온다.

 실행: python tools/measure_source_switch.py [표본수]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module import curve as CV           # noqa: E402
from module import dividend as DV        # noqa: E402
from module import features as F         # noqa: E402
from module import market as MK          # noqa: E402
from module import mc_engine as ME       # noqa: E402
from module import underlying as UL      # noqa: E402
from util import file_manager as fm      # noqa: E402

FACE = 10000
N_BASE, N_HIGH = 40_000, 100_000
IU = np.triu_indices(3, 1)


def ret_new(tk):
    """Underlying 가격 -> 로그수익률. module.mc.load_market 의 RET 과 같은 규약."""
    s = UL.price(tk)
    return None if s is None else np.log(s).diff()


def _inputs(rets, dt, cal):
    """(평균 σ, 상관행렬, 평균 ρ, 자산별 σ). 하나라도 못 만들면 전부 None."""
    sig = [F.vol_ewma_lam(r, dt) for r in rets]
    corr, _ = F.corr_bd(rets, dt, cal)
    if corr is None or np.isnan(sig).any():
        return None, None, None, None
    rho = float(np.mean([corr[a, b] for a, b in zip(*IU)]))
    return float(np.mean(sig)), corr, rho, sig


def measure(nprod=200):
    """요인별 영향 표. 지수전용과 개별회사포함을 절반씩 담는다."""
    df = MK.dataset()
    RET = MK.returns()
    cal = MK.calendar(RET)
    idx_only = df.udl1.str.startswith("^") & df.udl2.str.startswith("^") & df.udl3.str.startswith("^")
    rng = np.random.default_rng(0)
    want = {True: nprod // 2, False: nprod - nprod // 2}
    got = {True: 0, False: 0}
    rows = []
    for i in rng.permutation(len(df)):
        if len(rows) >= nprod:
            break
        r = df.iloc[int(i)]
        key = bool(idx_only.iloc[int(i)])
        if got[key] >= want[key]:
            continue
        tks = [r["udl1"], r["udl2"], r["udl3"]]
        ro = [RET.get(t) for t in tks]
        rn = [ret_new(t) for t in tks]
        if any(x is None for x in ro) or any(x is None for x in rn):
            continue
        dt = pd.Timestamp.fromordinal(int(r["isu_ord"]))
        dz = CV.zero_from_date(dt)
        if dz is None:
            continue
        so, co, rho_o, sv_o = _inputs(ro, dt, cal)
        sn, cn, rho_n, sv_n = _inputs(rn, dt, cal)
        if so is None or sn is None:
            continue
        qc = DV.qcont_lseg(tks, dt)
        if qc is None:
            continue
        divlog, _, _, _ = DV.build_div(tks, dt, float(r["tenor"]))
        k, p, lb, lp = MK.schedule(r)
        base = (float(r["B"]), k, float(r["tenor"]))
        kw = dict(seed=int(r["mc_seed"]), c=float(r["coupon"]), pmts=p, lz_barr=lb, lz_pmt=lp)

        def px(sig, corr, n, **dkw):
            return float(ME.mc_daily_t(sig, corr, dz, *base, n=n, **dkw, **kw)) * FACE

        rows.append(dict(
            item=r["item"], is_index_only=key,
            sig_old=so, sig_new=sn, rho_old=rho_o, rho_new=rho_n,
            mc_base=px(sv_o, co, N_BASE, divlog=divlog),
            mc_px=px(sv_n, cn, N_BASE, divlog=divlog),
            mc_div=px(sv_o, co, N_BASE, qcont=qc),
            mc_path=px(sv_o, co, N_HIGH, divlog=divlog),
            mc_all=px(sv_n, cn, N_HIGH, qcont=qc)))
        got[key] += 1
    return pd.DataFrame(rows)


def report(T):
    """요인별 변화를 지수전용 / 개별회사포함으로 나눠 출력."""
    pd.set_option("display.width", 200)
    print("표본 {}건 (지수전용 {} / 개별회사포함 {})".format(
        len(T), int(T.is_index_only.sum()), int((~T.is_index_only).sum())))
    print()
    print("=== 시장입력 변화 (가격계열 교체) ===")
    for lab, m in [("지수전용", T.is_index_only), ("개별회사포함", ~T.is_index_only)]:
        s = T[m]
        if not len(s):
            continue
        print("  {:12} σ 평균|차이| {:.5f} (최대 {:.5f}) | ρ 평균|차이| {:.5f} (최대 {:.5f})".format(
            lab, (s.sig_new - s.sig_old).abs().mean(), (s.sig_new - s.sig_old).abs().max(),
            (s.rho_new - s.rho_old).abs().mean(), (s.rho_new - s.rho_old).abs().max()))
    print()
    print("=== 요인별 MC 가격 변화 (기준 = 현행) ===")
    for lab, m in [("지수전용", T.is_index_only), ("개별회사포함", ~T.is_index_only), ("전체", T.item.notna())]:
        s = T[m]
        if not len(s):
            continue
        print("  [{}] n={}".format(lab, len(s)))
        for col, nm in [("mc_px", "가격계열만"), ("mc_div", "배당만"),
                        ("mc_path", "경로만 10만"), ("mc_all", "셋 다")]:
            e = s[col] - s.mc_base
            print("    {:12} 평균 {:+8.2f}원 | 평균|차이| {:7.2f}원 | 최대 {:8.2f}원 | 상관 {:.6f}".format(
                nm, e.mean(), e.abs().mean(), e.abs().max(),
                float(np.corrcoef(s.mc_base, s[col])[0, 1])))
    return T


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    T = measure(n)
    T.to_csv(fm.stat("source_switch_impact"), index=False, encoding="utf-8-sig")
    report(T)
    print()
    print("저장:", fm.stat("source_switch_impact").name)
