# -*- coding: utf-8 -*-
"""전체 els3_dataset MC — 두 변형 (노트북 5 엔진, 40k 경로, 공통 난수 seed = mc_seed). 노트북 17 의 전·후와 같은 정의.
 A  NS · HV120 · 상관 영업일 120 · q = 0
 B  부트스트랩(CD91 + IRS) · EWMA120(λ = 0.99) · 상관 영업일 120 · 이산배당
 정본(els3_dataset.parquet 의 mc)은 건드리지 않는다 → data/cache/mcvar/shard_<k>of<N>.parquet (500개마다 중간저장, 재개 가능).
 stage-1 입력도 함께 저장: 자산별 σ(A: HV120, B: EWMA120) · 쌍별 상관 · 자산별 배당수익률 q · 부트스트랩 제로금리(MSAMP 노드).
 배당: DAX(^GDAXI)는 총수익지수라 0, 삼성바이오로직스·LG에너지솔루션은 무배당(분배 기록 0)이라 자연히 0.

 실행 (프로젝트 루트):  python tools/mc_full_variants.py shard <k> <N>   |   bench [n]   |   combine <N>"""
import os, sys, time
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module import curve as CV
from module import dividend as DV
from module import features as F
from module import market as MK
from module import mc_engine as ME
from module.features import MSAMP
from util import file_manager as fm

OUT = fm.CACHE / "mcvar"; OUT.mkdir(exist_ok=True)
NPATH = 40_000               # 정본 mc_A/mc_B 의 경로 수
DIV = os.environ.get("MCVAR_DIV", "yf")      # 배당 소스: yf(이산, 정본) | lseg(연속)
TAG = "" if DIV == "yf" else "_" + DIV            # 샤드·산출물 이름을 소스별로 분리
OUT_NAME = "mc_variants{}.parquet".format(TAG)
RET = MK.returns()
KBD = MK.calendar(RET)


def price_row(i, r):
    dt = pd.Timestamp(date.fromordinal(int(r["isu_ord"]))); ts = [r["udl1"], r["udl2"], r["udl3"]]
    rets = [RET[t] for t in ts]
    s120 = [F.vol_win(x, dt, 120) for x in rets]
    e120 = [F.vol_ewma_lam(x, dt) for x in rets]
    c120, _ = F.corr_bd(rets, dt, KBD)
    dz_ns, mi = MK.ns_zero(dt), CV.market_inputs(dt)
    rec = dict(i=i, item=r["item"])
    if c120 is None or np.isnan(s120 + e120).any() or dz_ns is None or mi is None:
        rec["fail"] = "inputs"; return rec
    dz_bt = CV.curve_dz(*CV.bootstrap(mi["cd"], mi["swaps"]))
    strikes, pmts, lzb, lzp = MK.schedule(r); ten = float(r["tenor"]); B = float(r["B"])
    kw = dict(n=NPATH, seed=int(r["mc_seed"]), c=float(r["coupon"]), pmts=pmts, lz_barr=lzb, lz_pmt=lzp)
    if DIV == "lseg":                       # LSEG 는 배당 이벤트가 없어 연속 q 로만 쓸 수 있다
        qc = DV.qcont_lseg(ts, dt)
        if qc is None:
            rec["fail"] = "lseg div"; return rec
        divlog, nev = None, 0
        dkw = dict(qcont=qc)
    else:
        divlog, qc, nev, _ = DV.build_div(ts, dt, ten)
        dkw = dict(divlog=divlog)
    rec["mc_A"] = ME.mc_daily_t(s120, c120, dz_ns, B, strikes, ten, **kw)
    rec["mc_B"] = ME.mc_daily_t(e120, c120, dz_bt, B, strikes, ten, **dkw, **kw)
    iu = np.triu_indices(3, 1)
    rec.update({f"sA_{j+1}": s120[j] for j in range(3)}); rec.update({f"sB_{j+1}": e120[j] for j in range(3)})
    rec.update({f"rho_{a+1}{b+1}": float(c120[a, b]) for a, b in zip(*iu)})
    rec.update({f"q_{j+1}": float(qc[j]) for j in range(3)}); rec["n_div"] = int(nev)
    rec.update({f"bu{j}": float(v) for j, v in enumerate(dz_bt(MSAMP))})
    rec["r_bt"], rec["r_ns"] = float(dz_bt(ten)[0]), float(dz_ns(ten)[0])
    return rec


def run_shard(k, N, limit=None):
    df = MK.dataset()
    idx = np.arange(len(df))[k::N][:limit]
    outp, part = OUT / f"shard_{k}of{N}{TAG}.parquet", OUT / f"shard_{k}of{N}{TAG}.part.parquet"
    if outp.exists() and limit is None:
        print(f"shard {k}/{N}: 이미 완료", flush=True); return
    rows = pd.read_parquet(part).to_dict("records") if part.exists() and limit is None else []
    done = {int(x["i"]) for x in rows}
    t0 = time.time(); n0 = len(rows)
    for n_, i in enumerate(idx):
        if int(i) in done:
            continue
        rows.append(price_row(int(i), df.iloc[int(i)]))
        m = len(rows) - n0
        if m % 500 == 0 and limit is None:
            pd.DataFrame(rows).to_parquet(part)
            el = time.time() - t0
            print(f"shard {k}/{N}: {len(rows):,}/{len(idx):,}  {el/60:.1f}분  ETA {el/m*(len(idx)-len(rows))/60:.0f}분", flush=True)
    R = pd.DataFrame(rows)
    if limit is not None:
        return R, time.time() - t0
    R.to_parquet(outp)
    if part.exists():
        part.unlink()
    print(f"shard {k}/{N} DONE {len(R):,} in {(time.time()-t0)/60:.1f}분", flush=True)


def combine(N):
    R = pd.concat([pd.read_parquet(OUT / f"shard_{k}of{N}{TAG}.parquet") for k in range(N)]).sort_values("i").reset_index(drop=True)
    assert len(R) == len(MK.dataset()) and (R["item"].values == MK.dataset()["item"].values).all(), "행 불일치"
    R.to_parquet(OUT / OUT_NAME)
    print(f"combined {len(R):,} | fail {int(R.get('fail', pd.Series(dtype=object)).notna().sum())} | "
          f"mc_A {R.mc_A.mean():.4f} | mc_B {R.mc_B.mean():.4f} | fair {MK.dataset().fair.mean():.4f}")
    return R


STALE_DAYS = 14


def fix():
    """실패 행 재계산(IRS 14일 규칙 적용 후) + 가격 신선도 표시.
     stale_px = 기초자산 중 하나라도 발행일 전 최신 수익률이 STALE_DAYS 일보다 오래됨 (가격 캐시가 2025-06-30 에 끊긴 34종목 등).
     stale_px 상품은 σ·상관이 오래된 가격으로 계산되므로 비교에서 제외한다."""
    p = OUT / OUT_NAME; R = pd.read_parquet(p)
    bad = R.index[R["fail"].notna()] if "fail" in R else []
    for j in bad:
        rec = price_row(int(R.at[j, "i"]), MK.dataset().iloc[int(R.at[j, "i"])])
        for c, v in rec.items():
            R.at[j, c] = v
        if "fail" not in rec:
            R.at[j, "fail"] = None
    dts = [pd.Timestamp(date.fromordinal(int(o))) for o in MK.dataset().isu_ord]
    lag = np.zeros(len(R))
    for u in ("udl1", "udl2", "udl3"):
        lag = np.maximum(lag, [(d - RET[t][RET[t].index < d].index[-1]).days for t, d in zip(MK.dataset()[u], dts)])
    R["px_lag_days"] = lag.astype(int); R["stale_px"] = lag > STALE_DAYS
    R.to_parquet(p)
    ok = R["fail"].isna() & ~R["stale_px"]
    print(f"재계산 {len(bad)}개 → 남은 실패 {int(R['fail'].notna().sum())} | stale_px {int(R.stale_px.sum())} | "
          f"비교 사용 {int(ok.sum()):,}/{len(R):,} | 남은 실패 ⊂ stale_px: {bool((R['fail'].notna() <= R['stale_px']).all())}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if mode == "shard":
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "combine":
        combine(int(sys.argv[2]))
    elif mode == "fix":
        fix()
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        R, el = run_shard(0, 1000, limit=n)          # 전 기간에 흩어진 n 개
        F = 10000; fair = MK.dataset().fair.values[R.i.values]
        print(f"bench {len(R)}개 {el:.1f}s → {el/len(R):.2f}s/상품 | fair−A {np.mean(fair-R.mc_A)*F:+.0f}원 | fair−B {np.mean(fair-R.mc_B)*F:+.0f}원")
