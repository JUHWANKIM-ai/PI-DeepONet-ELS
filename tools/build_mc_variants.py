# -*- coding: utf-8 -*-
"""전체 els3_dataset MC — 두 변형 (노트북 5 엔진, 40k 경로, 공통 난수 seed = mc_seed). 노트북 17 의 전·후와 같은 정의.
 A  NS · HV120 · 상관 영업일 120 · q = 0
 B  부트스트랩(CD91 + IRS) · EWMA120(λ = 0.99) · 상관 영업일 120 · 이산배당
 정본(els3_dataset.parquet 의 mc)은 건드리지 않는다 → data/cache/mcvar/shard_<k>of<N>.parquet (500개마다 중간저장, 재개 가능).
 stage-1 입력도 함께 저장: 자산별 σ(A: HV120, B: EWMA120) · 쌍별 상관 · 자산별 배당수익률 q · 부트스트랩 제로금리(MSAMP 노드).
 배당: DAX(^GDAXI)는 총수익지수라 0, 삼성바이오로직스·LG에너지솔루션은 무배당(분배 기록 0)이라 자연히 0.

 1·2스타(els12_dataset)는 star12 모드가 mc_variants12 를 만든다 — 변형 B 만.

 실행 (프로젝트 루트):  python tools/build_mc_variants.py shard <k> <N>   |   bench [n]
                       |   combine <N>   |   fix   |   star12
 배당 소스는 MCVAR_DIV 환경변수 (yf 기본 = 이산, 정본 / lseg = 연속). 소스별로 파일명이 갈린다."""
import os, sys, time
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import module.mc as MCM
from module import curve as CV
from module import dividend as DV
from module import features as F
from module import market as MK
from module.seeds import mc_seed_for
from module import mc_engine as ME
from module.features import MSAMP
from util import file_manager as fm

OUT = fm.CACHE / "mcvar"; OUT.mkdir(exist_ok=True)
NPATH = 100_000              # 방법론 목표 경로 수 (40,000 대비 잡음 1.71배 감소, 편향 변화 없음)
DIV = os.environ.get("MCVAR_DIV", "lseg")    # 배당 소스: lseg(연속, 기본) | yf(이산, 비교용)
TAG = "" if DIV == "lseg" else "_" + DIV          # 기본값이 접미사 없는 이름 — 노트북이 그대로 읽는다
# ---------------- 출처 스탬프 ----------------
# `mc_variants.parquet` 는 이름이 그대로인 채 의미가 바뀐다 (yf·40k·58,790 -> lseg·100k·58,913).
# 파일 안에 무엇으로 만든 것인지 적어 두지 않으면 (1) 되돌림이 깨지고 (2) 데이터셋 갱신과
# (3) 재학습 판단이 "파일이 있는가" 에 의존해 낡은 산출물을 조용히 재사용한다.
# parquet **메타데이터**에 실어 보낸다 — 열을 늘리면 기존 소비자가 깨진다.
_STAMP_KEY = b"els_label_stamp"


STALE_KEYS = ("npath", "div", "px_source", "n_rows")   # 비교에 쓰는 키 (built_at 은 제외)


def stamp_differs(got, want):
    """두 스탬프가 다른가. **증명할 수 없으면 True** (= 낡은 것으로 본다).

     종전에는 같은 비교식이 네 곳에 복사돼 있었고, 스탬프가 없으면 전부 '최신' 으로 보고해
     스탬프 이전 라벨이 놓인 트리에서 모든 단계가 건너뛰어졌다 — 이 장치가 막으려던 실패다.
     가중치에 대해서는 '증명할 수 없으면 보수적으로 다시' 를 택했으므로 같은 기준을 쓴다."""
    if got is None or want is None:
        return True
    return any(str(got.get(k)) != str(want.get(k)) for k in STALE_KEYS)


def stamp(n_rows=None):
    """이 실행이 만든 라벨의 출처. 소비자는 이걸로 '낡음' 을 판단한다."""
    return {"npath": NPATH, "div": DIV,
            "px_source": os.environ.get("PX_SOURCE", MCM.PX_SOURCES[1]),
            "n_rows": -1 if n_rows is None else int(n_rows),
            "built_at": date.today().isoformat()}


def write_labeled(df, path, n_rows=None):
    """라벨 DataFrame 을 출처 스탬프와 함께 쓴다."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta[_STAMP_KEY] = json.dumps(stamp(len(df) if n_rows is None else n_rows)).encode()
    pq.write_table(t.replace_schema_metadata(meta), str(path))


def read_stamp(path):
    """저장된 출처. 스탬프가 없으면 None (스탬프 도입 이전 파일)."""
    import json
    import pyarrow.parquet as pq
    path = Path(str(path))
    if not path.exists():
        return None
    md = pq.read_schema(str(path)).metadata or {}
    raw = md.get(_STAMP_KEY)
    return json.loads(raw.decode()) if raw else None




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
    got = DV.div_kwargs(ts, dt, ten, DIV)   # 소스 분기는 module.dividend 한 곳에서만
    if got is None:
        rec["fail"] = "lseg div"; return rec
    dkw, qc, nev = got
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
    write_labeled(R, outp)
    if part.exists():
        part.unlink()
    print(f"shard {k}/{N} DONE {len(R):,} in {(time.time()-t0)/60:.1f}분", flush=True)


def ensure_fail(R):
    """`fail` 열을 보장한다. price_row 는 실패했을 때만 그 열을 넣으므로, 전건 성공하면
     열이 아예 없는 프레임이 나오고 뒤쪽 `R["fail"].isna()` 가 KeyError 로 죽는다.

     라벨 스키마가 **실패 유무에 따라 달라지면 안 된다**. 실제로 전건 성공한 59,037건에서
     137분짜리 산출 직후에 터졌다 — 성공할수록 깨지는 코드였다."""
    if "fail" not in R.columns:
        R["fail"] = pd.Series([None] * len(R), index=R.index, dtype=object)
    return R


def combine(N):
    R = pd.concat([pd.read_parquet(OUT / f"shard_{k}of{N}{TAG}.parquet") for k in range(N)]).sort_values("i").reset_index(drop=True)
    assert len(R) == len(MK.dataset()) and (R["item"].values == MK.dataset()["item"].values).all(), "행 불일치"
    R = ensure_fail(R)
    write_labeled(R, OUT / OUT_NAME)
    print(f"combined {len(R):,} | fail {int(R.get('fail', pd.Series(dtype=object)).notna().sum())} | "
          f"mc_A {R.mc_A.mean():.4f} | mc_B {R.mc_B.mean():.4f} | fair {MK.dataset().fair.mean():.4f}")
    return R


# ---------------- 1·2스타 (els12_dataset) ----------------
# 종전 생성기는 scratch/mc_12star.py 에만 있어 scratch 정리 때 사라졌고,
# mc_variants12.parquet 은 생성기 없는 라벨 파일로 남아 있었다. 이 모드가 그 파일을 재현한다.
# 3스타와 달리 변형 A(NS 커브)는 만들지 않는다 — module/star.py 는 변형 B 라벨만 쓴다.
# 중복 슬롯 쌍의 상관은 정확히 1.0 으로 둔다 (corr_bd 가 0.999… 로 자르는 것을 되돌린다).

OUT12_NAME = "mc_variants12{}.parquet".format(TAG)


def dataset12():
    """els12_dataset (발행일 정렬) + 행번호 seed. mc_seed 열이 원천에 없어 여기서 만든다."""
    d = pd.read_parquet(fm.DATA / "els12_dataset.parquet").sort_values("isu_ord").reset_index(drop=True)
    d["mc_seed"] = mc_seed_for(d["item"])       # 상품 내용에서 유도 (행 번호 아님)
    return d


def dup_corr(C, ts):
    """중복 슬롯 쌍의 상관을 1.0 으로 복원. tools/mc_check_12star.exact_dup_corr 와 같은 규칙."""
    C = np.array(C, float, copy=True)
    for a in range(3):
        for b in range(3):
            if a != b and ts[a] == ts[b]:
                C[a, b] = 1.0
    np.fill_diagonal(C, 1.0)
    return C


def price_row12(i, r):
    dt = pd.Timestamp(date.fromordinal(int(r["isu_ord"]))); ts = [r["udl1"], r["udl2"], r["udl3"]]
    rets = [RET.get(t) for t in ts]
    rec = dict(i=i, item=r["item"], nstar=int(r["nstar"]))
    if any(x is None for x in rets):
        rec["fail"] = "inputs"; return rec
    e120 = [F.vol_ewma_lam(x, dt) for x in rets]
    c120, _ = F.corr_bd(rets, dt, KBD)
    mi = CV.market_inputs(dt)
    if c120 is None or np.isnan(e120).any() or mi is None:
        rec["fail"] = "inputs"; return rec
    c120 = dup_corr(c120, ts)
    dz_bt = CV.curve_dz(*CV.bootstrap(mi["cd"], mi["swaps"]))
    strikes, pmts, lzb, lzp = MK.schedule(r); ten = float(r["tenor"])
    kw = dict(n=NPATH, seed=int(r["mc_seed"]), c=float(r["coupon"]), pmts=pmts, lz_barr=lzb, lz_pmt=lzp)
    got = DV.div_kwargs(ts, dt, ten, DIV)   # 소스 분기는 module.dividend 한 곳에서만
    if got is None:
        rec["fail"] = "lseg div"; return rec
    dkw, qc, nev = got
    rec["mc_B"] = ME.mc_daily_t(e120, c120, dz_bt, float(r["B"]), strikes, ten, **dkw, **kw)
    iu = np.triu_indices(3, 1)
    rec.update({f"sB_{j+1}": e120[j] for j in range(3)})
    rec.update({f"rho_{a+1}{b+1}": float(c120[a, b]) for a, b in zip(*iu)})
    rec.update({f"q_{j+1}": float(qc[j]) for j in range(3)}); rec["n_div"] = int(nev)
    rec.update({f"bu{j}": float(v) for j, v in enumerate(dz_bt(MSAMP))})
    rec["r_bt"] = float(dz_bt(ten)[0])
    return rec


def run12(limit=None):
    d = dataset12()
    idx = np.arange(len(d))[:limit]
    outp, part = OUT / OUT12_NAME, OUT / "star12{}.part.parquet".format(TAG)
    rows = pd.read_parquet(part).to_dict("records") if part.exists() and limit is None else []
    done = {int(x["i"]) for x in rows}
    t0 = time.time(); n0 = len(rows)
    for i in idx:
        if int(i) in done:
            continue
        rows.append(price_row12(int(i), d.iloc[int(i)]))
        m = len(rows) - n0
        if m % 500 == 0 and limit is None:
            pd.DataFrame(rows).to_parquet(part)
            el = time.time() - t0
            print(f"star12: {len(rows):,}/{len(idx):,}  {el/60:.1f}분  ETA {el/m*(len(idx)-len(rows))/60:.0f}분", flush=True)
    R = pd.DataFrame(rows).sort_values("i").reset_index(drop=True)
    if limit is not None:
        return R, time.time() - t0
    write_labeled(R, outp)
    if part.exists():
        part.unlink()
    print(f"star12 DONE {len(R):,} | fail {int(R.get('fail', pd.Series(dtype=object)).notna().sum())} | "
          f"mc_B {R.mc_B.mean():.4f} in {(time.time()-t0)/60:.1f}분")
    return R


STALE_DAYS = 14


def fix():
    """실패 행 재계산(IRS 14일 규칙 적용 후) + 가격 신선도 표시.
     stale_px = 기초자산 중 하나라도 발행일 전 최신 수익률이 STALE_DAYS 일보다 오래됨 (가격 캐시가 2025-06-30 에 끊긴 34종목 등).
     stale_px 상품은 σ·상관이 오래된 가격으로 계산되므로 비교에서 제외한다."""
    p = OUT / OUT_NAME; R = ensure_fail(pd.read_parquet(p))
    bad = R.index[R["fail"].notna()]
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
    ok = R["fail"].isna() & ~R["stale_px"]
    # **마지막에 한 번만 쓴다.** 종전에는 여기보다 먼저 써서, 다음 줄에서 죽었을 때
    # stale_px 는 있고 fail 은 없는 반쪽 파일이 남았다. 그 파일을 '완료' 로 오판하면
    # 미완성 라벨이 그대로 하류로 흘러간다.
    write_labeled(R, p)
    print(f"재계산 {len(bad)}개 → 남은 실패 {int(R['fail'].notna().sum())} | stale_px {int(R.stale_px.sum())} | "
          f"비교 사용 {int(ok.sum()):,}/{len(R):,} | 남은 실패 ⊂ stale_px: {bool((R['fail'].notna() <= R['stale_px']).all())}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if mode == "shard":
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "combine":
        combine(int(sys.argv[2]))
    elif mode == "star12":
        run12()
    elif mode == "fix":
        fix()
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        R, el = run_shard(0, 1000, limit=n)          # 전 기간에 흩어진 n 개
        F = 10000; fair = MK.dataset().fair.values[R.i.values]
        print(f"bench {len(R)}개 {el:.1f}s → {el/len(R):.2f}s/상품 | fair−A {np.mean(fair-R.mc_A)*F:+.0f}원 | fair−B {np.mean(fair-R.mc_B)*F:+.0f}원")


def _mark_path(path):
    return Path(str(path) + ".stamp.json")


def mark_step(path):
    """그 산출물이 **어떤 라벨로** 만들어졌는지 남긴다 (`<path>.stamp.json`).

     노트북 1 의 단계들은 전부 mc_seed 를 쓰므로 라벨이 바뀌면 같이 낡는다. 산출물마다
     다른 완료 표식을 고르면(열 존재, 전부 NaN, 파일 존재 ...) 다음 산출물에서 또 틀린다 —
     이 세션에서 그 실수를 네 번 했다. 표식을 하나로 통일한다."""
    import json
    st = read_stamp(OUT / OUT_NAME)
    _mark_path(path).write_text(json.dumps(st or {}, ensure_ascii=False), encoding="utf-8")


def step_stale(path):
    """지금 라벨로 그 단계를 다시 돌려야 하는가. 없거나 표식이 없거나 다르면 True."""
    import json
    p = Path(str(path))
    if not p.exists():
        return True
    m = _mark_path(p)
    if not m.exists():
        return True
    try:
        got = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return True
    return stamp_differs(got, read_stamp(OUT / OUT_NAME))
