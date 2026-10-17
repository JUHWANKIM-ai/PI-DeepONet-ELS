# -*- coding: utf-8 -*-
"""1-star / 2-star ELS 데이터셋 빌드 — 기존 3-star 파이프라인을 그대로 쓰기 위한 **중복 패딩**.

 현행 엔진·피처·망은 전부 3자산 전제다 (min(S1,S2,S3), sig1..3, rho12/13/23).
 자산 수를 일반화하는 대신 **슬롯을 중복으로 채운다**:
     1-star : (A, A, A)   상관 1
     2-star : (A, B, B)   rho_AB = 실측, rho_BB = 1
 worst-of 는 min 이므로 중복 슬롯은 값에 영향이 없고, MC 엔진의 chol_psd 가 특이행렬을
 허용한다(검증: 복원오차 1e-8, 중복축 잔여 비상관 1.4e-4 -> 3년 sigma20% 에서 가격영향 5e-5).

 module.build_source 와 **같은 스키마**를 만든다 (+ nstar 열).
 산출: data/els12_dataset.parquet
 실행: python tools/build_12star.py
"""
import sys, json, time, bisect
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from module import features as F
from module import mc as MCM
from module.build_source import (STRUCTS, FAIR_LO, FAIR_HI, TEN_LO, TEN_HI,
                                 encode_barrier, needs_linear_pmt, _safe)
from module.schedule import load_schedules, NSTRK, SCHED_COLS
from util import file_manager as fm


def pad3(ts):
    """기초자산 티커를 3슬롯으로 중복 패딩. 1개 -> AAA, 2개 -> ABB."""
    if len(ts) == 1:
        return [ts[0]] * 3
    if len(ts) == 2:
        return [ts[0], ts[1], ts[1]]
    return list(ts)[:3]


def mom6m(rets, dt, win=126):
    vals = []
    for r in rets:
        if r is None:
            return 0.0
        w = r[r.index < dt].tail(win)
        if len(w) < 40:
            return 0.0
        vals.append(float(w.sum()))
    return float(np.mean(vals)) if vals else 0.0


def main():
    """1·2스타 데이터셋을 만들어 data/els12_dataset.parquet 에 저장한다. 노트북에서도 부를 수 있다."""
    CA, RAW = fm.CACHE, fm.RAW
    mapping = json.loads((fm.RAW / "udly_ticker_map.json").read_text(encoding="utf-8"))
    RET = MCM.returns_by_ticker(mapping)        # 가격 적재는 module.mc 한 곳에서만
    KRW = pd.read_parquet(fm.rate("krw_curve.parquet"))[["call", "m3", "y10"]]

    ac = pd.read_csv(RAW / "LAKE_V2_DART_AUTO_CALL.csv", low_memory=False)
    sc = pd.read_csv(RAW / "LAKE_V2_DART_SCHD_INFO.csv", low_memory=False)
    ud = pd.read_csv(RAW / "LAKE_V2_DART_UDLY_INFO.csv", low_memory=False)
    for c in ("ISU_DT", "MAT_DT", "SUB_START_DT", "SUB_END_DT"):
        ac[c] = pd.to_datetime(ac[c], errors="coerce")
    ac["tenor"] = (ac["MAT_DT"] - ac["ISU_DT"]).dt.days / 365.25
    ac["fair"] = ac["FAIR_VALUE"] / ac["ISU_PRC_DETAIL"]
    nu = ud.groupby("ITEM_CD")["UDLY_ID"].nunique()
    sel = nu[nu.isin([1, 2])].index
    umap = ud[ud.ITEM_CD.isin(sel)].groupby("ITEM_CD")["UDLY_ID"].apply(list)
    nstar_of = nu.to_dict()
    SCHED = load_schedules()
    barr_pct = sc[sc.SCHD_TYPE == 1].groupby("ITEM_CD")["BARR_1"].min()
    struct = pd.Series(False, index=ac.index)
    for o, k in STRUCTS:
        struct |= (ac["OPT_TYPE"].eq(o) & ac["KNCK_IN_YN"].eq(k))
    cand = ac[ac["ITEM_CD"].isin(sel) & ac["CUR_CD"].eq("KRW") & struct
              & ac["fair"].between(FAIR_LO, FAIR_HI) & ac["tenor"].between(TEN_LO, TEN_HI)]
    print("1·2스타 후보 {:,} (1스타 {:,} / 2스타 {:,})".format(
        len(cand), int(cand.ITEM_CD.map(nstar_of).eq(1).sum()),
        int(cand.ITEM_CD.map(nstar_of).eq(2).sum())), flush=True)

    rows, drops, t0 = [], {}, time.time()
    for it, rw in cand.set_index("ITEM_CD").iterrows():
        try:
            raw_u = umap.get(it, [])
            ts0 = [mapping.get(x) for x in raw_u]
            if not ts0 or any(t is None for t in ts0):
                drops["티커 매핑 실패"] = drops.get("티커 매핑 실패", 0) + 1; continue
            ts = pad3(sorted(set(ts0)) if len(set(ts0)) < len(ts0) else ts0)
            ns = len(set(ts))
            rets = [RET.get(t) for t in ts]
            if any(r is None for r in rets):
                drops["가격이력 없음"] = drops.get("가격이력 없음", 0) + 1; continue
            dt = rw.ISU_DT
            sigs = [F.vol180(r, dt) for r in rets]
            if any(pd.isna(sigs)):
                drops["180일 변동성 부족"] = drops.get("180일 변동성 부족", 0) + 1; continue
            corr = F.corr180(rets, dt)
            if corr is None:
                drops["180일 상관 부족"] = drops.get("180일 상관 부족", 0) + 1; continue
            if it not in SCHED.index or not bool(SCHED.at[it, "sched_ok"]):
                why = SCHED.at[it, "sched_drop"] if it in SCHED.index else "스케줄 없음"
                drops["스케줄: {}".format(why)] = drops.get("스케줄: {}".format(why), 0) + 1; continue
            if bool(SCHED.at[it, "lz_from_prev"]):
                drops["리자드 FROM_PREV"] = drops.get("리자드 FROM_PREV", 0) + 1; continue
            srow = SCHED.loc[it]
            nobs_ = int(srow["nobs"])
            strikes = [float(srow["strk_{}".format(j)]) for j in range(nobs_)]
            B = encode_barrier(rw.KNCK_IN_YN, barr_pct.get(it, np.nan))
            c = rw.ANL_RTRN / 100; ten = rw.tenor
            if any(pd.isna(x) for x in [B, c, ten]):
                drops["배리어/쿠폰/만기 결측"] = drops.get("배리어/쿠폰/만기 결측", 0) + 1; continue
            beta = F.krw_beta(KRW.asof(dt).values)
            if beta is None:
                continue
            u = F.krw_curve_nodes(beta); r = float(F.zero_curve(beta, np.array([ten]))[0])
            iu = np.triu_indices(3, 1); rr = np.sort(np.clip(corr[iu], -0.999, 0.999))
            rho12, rho13, rho23 = float(rr[0]), float(rr[1]), float(rr[2])
            rho_v = float(np.mean(rr))
            ss = np.sort(sigs); sig1, sig2, sig3 = float(ss[0]), float(ss[1]), float(ss[2])
            smean = float(np.mean(sigs)); sig_eff = smean * float(np.sqrt(1.0 + max(0.0, 1.0 - rho_v)))
            Klast = float(strikes[-1]); Kfst = float(strikes[0])
            sd = ((rw.SUB_END_DT - rw.SUB_START_DT).days
                  if pd.notna(rw.SUB_END_DT) and pd.notna(rw.SUB_START_DT) else np.nan)
            rec = dict(item=it, nstar=ns, sig1=sig1, sig2=sig2, sig3=sig3,
                       rho12=rho12, rho13=rho13, rho23=rho23,
                       sig_mean=smean, sig_max=sig3, sig_min=sig1, rho=rho_v, sig_eff=sig_eff,
                       cpn_spread=float(c - r), b_over_k=float(B / Klast) if Klast > 0 else 1.0,
                       stepdown=float(Kfst - Klast), mom6m=mom6m(rets, dt), isu_ord=int(dt.toordinal()),
                       r=r, B=float(B), Kfirst=Kfst, K=Klast, coupon=float(c), tenor=float(ten),
                       nobs=nobs_, fair=float(rw.fair), issuer=str(rw.ISU_ORG),
                       risk=str(rw.RISK_GRADE), ptype=str(rw.PRODUCT_TYPE), rdmp=str(rw.RDMP_TYPE),
                       imonth=str(int(dt.month)),
                       amt=float(rw.ACT_ISU_AMT) if pd.notna(rw.ACT_ISU_AMT) else np.nan,
                       sbrt=float(rw.SB_RT) if pd.notna(rw.SB_RT) else np.nan,
                       dvrt=float(rw.DV_RT) if pd.notna(rw.DV_RT) else np.nan,
                       prcp=float(rw.PRCP_GRTE_RT) if pd.notna(rw.PRCP_GRTE_RT) else np.nan,
                       kigrc=float(rw.KNCK_IN_GRC_PRD) if pd.notna(rw.KNCK_IN_GRC_PRD) else np.nan,
                       iyear=float(dt.year), subdays=float(sd))
            for j in range(10):
                rec["u{}".format(j)] = float(u[j])
            rec["curve_level"] = float(u.mean()); rec["curve_slope"] = float(u[9] - u[0])
            rec["curve_curv"] = float(2 * u[4] - u[0] - u[9])
            rec["opt_type"] = str(rw.OPT_TYPE)
            rec["ki_yn"] = int(rw.KNCK_IN_YN)
            ts_sorted = sorted(ts)
            rec["udl1"], rec["udl2"], rec["udl3"] = ts_sorted
            rec["udl_key"] = "|".join(ts_sorted)
            for cname in SCHED_COLS:
                rec[cname] = float(srow[cname]) if pd.notna(srow[cname]) else np.nan
            rec["pmt_linear"] = int(needs_linear_pmt(
                [rec["pmt_{}".format(q)] for q in range(nobs_)], c, ten, bool(srow["pmt_nonmono"])))
            if rec["pmt_linear"]:
                Nd = int(round(ten * 365))
                od = np.clip(np.round(np.arange(1, nobs_ + 1) * (ten / nobs_) * 365).astype(int), 1, Nd)
                for j, v in enumerate(c * (od / 365.0)):
                    rec["pmt_{}".format(j)] = float(v)
                for j in range(nobs_, NSTRK):
                    rec["pmt_{}".format(j)] = rec["pmt_{}".format(nobs_ - 1)]
            rows.append(rec)
        except Exception as e:
            k = "예외: {}".format(type(e).__name__)
            drops[k] = drops.get(k, 0) + 1
            continue
    df = pd.DataFrame(rows).sort_values("isu_ord").reset_index(drop=True)
    print("built {:,} products in {:.0f}s".format(len(df), time.time() - t0))
    print("  스타별:", df.groupby("nstar").size().to_dict())
    print("  구조별:", df.groupby(["opt_type", "ki_yn"]).size().to_dict())
    if drops:
        print("  탈락 사유:", dict(sorted(drops.items(), key=lambda x: -x[1])))
    o = df["isu_ord"].tolist(); sm = df["sig_mean"].values; rmv = np.zeros(len(df))
    for i in range(len(df)):
        hi = bisect.bisect_left(o, o[i]); lo = bisect.bisect_left(o, o[i] - 90)
        rmv[i] = sm[lo:hi].mean() if hi > lo else (sm[:hi].mean() if hi > 0 else sm[i])
    df["recent_mktvol"] = rmv.astype("float32")
    out = fm.DATA / "els12_dataset.parquet"
    df.to_parquet(out)
    print("saved", out, "rows", len(df))


if __name__ == "__main__":
    main()
