# -*- coding: utf-8 -*-
"""MC 이론가 엔진 (미래에셋 36449 방식) — 노트북(1_MC_pricer)·병렬 샤드 스크립트 공용(DRY).
일별 GBM + 촐레스키 기하 worst-of + 일별 KI. **시장입력 정의는 호출부가 정한다** —
정본 라벨(`mc_B`)은 EWMA120(λ=0.99) · KRX 영업일 120일 쌍별 상관 · CD+IRS 부트스트랩 커브 ·
연속 배당수익률 q · 100,000 경로이고, `generate/build_mc_variants.py` 가 조립한다.
이 모듈이 맡는 것은 **가격 계열 적재**(returns_by_ticker)와 데이터셋 조립뿐이다.
곡선·vol·corr 정의는 module.features 사용.

*** 이 모듈(numpy)은 프로덕션 재산출에서 퇴역했다. 페이오프 규칙의 정본은 module/mc_engine.py (torch) 다.
    여기 mc_daily 는 레거시 규칙(선형 쿠폰·리자드 없음·낙인 판정)만 지원하며, 기존 데이터셋 mc 재현과
    엔진 대조(scratch/verify_engine_agreement.py)에만 쓴다. 새 규칙을 여기에 추가하지 말 것. ***

캘리브레이션(선택): 역사 변동성은 내재변동성보다 낮아 MC가 공정가치를 상회한다. 관측 입력(sig_eff)의
연속 함수 k(sig_eff)=piecewise-linear(=data/cache/calib_kmap.json)를 σ에 곱해 편향을 줄인다(불연속 없음).
kmap=None 이면 미보정(pre-calibration) 버전 = 원래 MC.
"""
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from util import file_manager as fm
from . import features as F

NPATH = 100_000
PCHUNK = 2_000
MC_COLS = [
    "mc",
    "mc_vol1",
    "mc_vol2",
    "mc_vol3",
    "mc_rho12",
    "mc_rho13",
    "mc_rho23",
    "mc_r_krw",
    "mc_k",
]


def _safe(t):
    return "px_" + t.replace("^", "_").replace(".", "_") + ".parquet"


PX_SOURCES = ("cache", "underlying")


def returns_by_ticker(mapping):
    """기초자산 ID->티커 매핑 -> {티커: 로그수익률 Series}. **가격 적재의 유일한 출처.**

    공급원은 환경변수 PX_SOURCE 로 고른다.
      underlying  share_underlyings_v2 의 Close_Adj  액면분할만 조정 — 기본값
      cache       data/cache/px_*.parquet            배당 역조정(총수익) — 종전 라벨 재현용

    종전에는 같은 로직이 load_market · build_source.load_inputs · build_12star.main
    세 곳에 복사돼 있었고, 원천 전용 실행에서 한 곳씩 차례로 터졌다."""
    src = os.environ.get("PX_SOURCE", "underlying")
    if src not in PX_SOURCES:
        raise ValueError(
            "PX_SOURCE 는 {} 중 하나여야 한다 (받은 값: {!r})".format(" | ".join(PX_SOURCES), src)
        )
    out = {}
    for uid, t in mapping.items():
        if not t or t in out:
            continue
        if src == "underlying":
            from . import underlying as UL

            # UDLY_ID 를 대체 키로 넘긴다. LSEG 가 yfinance 티커를 주지 않는 기초자산이 있고
            # (^SX7E) master.csv 의 키는 UDLY_ID 다 — 안 넘기면 그 상품들이 조용히 사라진다.
            u = UL.price(t, udly=uid)
            if u is not None:
                out[t] = np.log(u).diff()
            continue
        f = fm.CACHE / _safe(t)
        if not f.exists():
            continue
        c = pd.read_parquet(f)["close"].dropna()
        out[t] = np.log(c[~c.index.duplicated()]).diff()
    return out


def load_market():
    """cache/raw → 시장데이터 dict(RET 로그수익률, KRW 곡선, 상품→기초자산·행사가 매핑).

    가격 공급원은 환경변수 PX_SOURCE 로 고른다.
      underlying  share_underlyings_v2 의 Close_Adj  액면분할만 조정 — **기본값**
      cache       data/cache/px_*.parquet   배당 역조정(총수익) — 종전 라벨 재현용
    원천 전용 재현이 기본 경로다. 저장된 mc_A/mc_B 를 비트 단위로 재현하려면
    scratch 백업을 되돌리고 PX_SOURCE=cache 로 돌려야 한다."""
    CA, RAW = fm.CACHE, fm.RAW
    mapping = json.loads((fm.RAW / "udly_ticker_map.json").read_text(encoding="utf-8"))
    RET = returns_by_ticker(mapping)
    KRW = pd.read_parquet(fm.rate("krw_curve.parquet"))[["call", "m3", "y10"]]
    if not (CA / "_u3map.json").exists() or not (CA / "_strk_by.json").exists():
        ud = pd.read_csv(RAW / "LAKE_V2_DART_UDLY_INFO.csv", low_memory=False)
        sc = pd.read_csv(RAW / "LAKE_V2_DART_SCHD_INFO.csv", low_memory=False)
        (CA / "_u3map.json").write_text(
            json.dumps(ud.groupby("ITEM_CD")["UDLY_ID"].apply(list).to_dict()), encoding="utf-8"
        )
        sc1 = sc[sc.SCHD_TYPE == 1].sort_values(["ITEM_CD", "SEQ"])
        (CA / "_strk_by.json").write_text(
            json.dumps(
                sc1.groupby("ITEM_CD")["STRK_1"]
                .apply(lambda s: [float(x) / 100 for x in s])
                .to_dict()
            ),
            encoding="utf-8",
        )
    return dict(
        mapping=mapping,
        RET=RET,
        KRW=KRW,
        u3map=json.loads((CA / "_u3map.json").read_text(encoding="utf-8")),
        strk_by=json.loads((CA / "_strk_by.json").read_text(encoding="utf-8")),
    )


def load_kmap(path=None):
    """캘리브레이션 매핑 로딩. 파일 없으면 None(미보정)."""
    p = fm.CACHE / "calib_kmap.json" if path is None else path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def calib_k(sig_eff, kmap):
    """관측 입력(sig_eff) → vol 프리미엄 배수 k. 연속 piecewise-linear 보간(knot 밖은 끝값으로 클램프).
    kmap=None 이면 1.0(미보정). (버킷 계단 버전은 불연속으로 DeepONet 앵커 학습을 해쳐 연속형으로 대체.)"""
    if kmap is None:
        return 1.0
    return float(np.interp(float(sig_eff), kmap["x"], kmap["y"]))


def mc_daily(sigs, corr, beta, B, strikes, c, ten, n=NPATH, seed=0, chunk=PCHUNK):
    """미래에셋 방식 일별 MC. 비연속-out cumsum 누수 회피 위해 연속 exp(cumsum) 사용."""
    nobs = len(strikes)
    N = int(round(ten * 365))
    dt = 1 / 365
    obs_day = np.clip(np.round(np.arange(1, nobs + 1) * (ten / nobs) * 365).astype(int), 1, N)
    obs_t = obs_day / 365.0
    times = np.arange(1, N + 1) * dt
    DF = np.exp(-F.zero_curve(beta, times) * times)
    lnDF = np.concatenate([[0.0], np.log(DF)])
    fdt = (-(np.diff(lnDF))).astype(np.float32)
    L = F.chol_psd(np.asarray(corr)).astype(np.float32)
    sig = np.asarray(sigs, np.float32)
    dtf = np.float32(dt)
    sq = np.float32(np.sqrt(dt))
    drift = fdt[:, None] - (0.5 * sig**2 * dtf)[None, :]
    rng = np.random.default_rng(seed)
    tot = 0.0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        Z = rng.standard_normal((m, N, 3), dtype=np.float32) @ L.T
        incr = drift[None, :, :] + sig[None, None, :] * sq * Z
        R = np.exp(np.cumsum(incr, axis=1))
        WP = R.min(axis=2)
        ki = (WP < B).any(axis=1)
        alive = np.ones(m, bool)
        pay = np.zeros(m)
        for j in range(nobs):
            oi = obs_day[j]
            hit = alive & (WP[:, oi - 1] >= strikes[j])
            pay[hit] = (1 + c * obs_t[j]) * DF[oi - 1]
            alive[hit] = False
        wT = WP[alive, -1]
        kA = ki[alive]
        pay[alive] = np.where(~kA, 1.0, wT) * DF[-1]
        tot += pay.sum()
        done += m
        del Z, incr, R, WP
    return tot / n


def prepare_one(mk, kmap, item, iord, ten, sig_eff, strikes=None):
    """상품 -> 엔진 공통 입력. numpy(price_one)와 torch(mc_engine.price_one_t)가 공유한다.
    sigs 는 캘리브레이션 배수 k 가 곱해진 값, audit 은 미보정 관측값(데이터셋 mc_vol*/mc_rho* 용)."""
    dt = pd.Timestamp(date.fromordinal(int(iord)))
    beta = F.krw_beta(mk["KRW"].asof(dt).values)
    strikes = list(strikes) if strikes is not None else mk["strk_by"].get(item)
    if beta is None or not strikes:
        return None
    rkrw = float(F.zero_curve(beta, np.array([ten]))[0])
    k = calib_k(sig_eff, kmap)
    ts = [mk["mapping"].get(x) for x in mk["u3map"].get(item, [])]
    ts = [t for t in ts if t]
    rets = [mk["RET"].get(t) for t in ts]
    if len(ts) == 3 and all(r is not None for r in rets):
        sigs = [F.vol180(r, dt) for r in rets]
        corr = F.corr180(rets, dt)
        if corr is not None and not any(pd.isna(sigs)):
            return dict(
                beta=beta,
                strikes=strikes,
                sigs=[k * s for s in sigs],
                corr=corr,
                rkrw=rkrw,
                k=k,
                audit=(sigs[0], sigs[1], sigs[2], corr[0, 1], corr[0, 2], corr[1, 2]),
            )
    return dict(
        beta=beta,
        strikes=strikes,
        sigs=[k * sig_eff] * 3,
        corr=np.eye(3),
        rkrw=rkrw,
        k=k,
        audit=(np.nan,) * 6,
    )  # 폴백: 단일자산 근사


def price_one(mk, kmap, item, iord, B, c, ten, sig_eff, n=NPATH, seed=0, strikes=None):
    """상품 하나 (레거시 numpy 엔진): (mc, vol1..3, rho12/13/23, r_krw, k) 9개.
    vol1..3 은 관측(미보정) 역사변동성이며 MC 내부에서는 k·σ 를 사용(k=적용배수)."""
    P = prepare_one(mk, kmap, item, iord, ten, sig_eff, strikes=strikes)
    if P is None:
        return (np.nan,) * len(MC_COLS)
    mc = mc_daily(P["sigs"], P["corr"], P["beta"], B, P["strikes"], c, ten, n=n, seed=seed)
    return (mc, *P["audit"], P["rkrw"], P["k"])
