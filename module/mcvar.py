# -*- coding: utf-8 -*-
"""MC 변형 번들 — 변형 A(NS · HV180 · q=0) vs **변형 B = 현행 방법론**(부트스트랩 · EWMA120 · 이산배당).

 `module.data.load` 가 돌려주는 정본 D 를 복사해 그 변형의 MC 입력으로 갈아끼운다.
 각 변형의 stage-1 은 **그 변형 MC 의 대리모형**이므로 입력도 그 MC 의 입력에 맞춰야 한다.

   VC    = [sigma 오름차순 3 | 상관 오름차순 3 | sig_eff]  (A: HV180, B: EWMA120)
           B 는 자산별 배당수익률 q 3개를 sigma 와 같은 순서로 덧붙여 **10열**이 된다
   CURVE = A: NS 노드 u0..u9, B: 부트스트랩 제로금리 (같은 MSAMP 노드)
   CON   = 계약 51개 (공통)
   rm    = 그 변형 MC 로 다시 계산한 recent_margin (인과, 구조별 90일)

 ⚠ 알려진 스키마 불일치: 변형 B 의 D.VC 는 10열이고 `module/data.py` 가 문서화한 7열과 다르다.
 정본 D 와 변형 D 가 같은 이름의 다른 타입이라는 뜻이다. data.load 에 variant 인자를 넣어
 한 종류로 합치는 것이 옳지만 기존 호출부 영향이 커서 별도 과제로 남겼다.

 라벨 원천은 `data/cache/mcvar/mc_variants.parquet` 이고 `generate/build_mc_variants.py` 가 만든다.
 종전에는 scratch/stage1_mc_variants.py 에만 있어 추적 파이프라인에서 쓸 수 없었다.
"""
import copy
import os

import numpy as np
import pandas as pd
import yaml

from util import file_manager as fm
from . import data as MD
from .mc_shards import recent_margin

FACE = 10000
_ST = None


def vc_block(s, rho, q=None):
    """[sigma 오름차순 3 | 상관 오름차순 3 | sig_eff] (+ q 3개). sigma 정렬에 q 도 맞춰 따라간다."""
    order = np.argsort(s, axis=1)
    ss = np.take_along_axis(s, order, 1)
    rr = np.sort(np.clip(rho, -0.999, 0.999), axis=1)
    seff = s.mean(1) * np.sqrt(1.0 + np.maximum(0.0, 1.0 - rr.mean(1)))
    cols = [ss, rr, seff[:, None]] + ([np.take_along_axis(q, order, 1)] if q is not None else [])
    return np.concatenate(cols, axis=1).astype("float32")


DIV = os.environ.get("MCVAR_DIV", "lseg")        # 배당 소스: lseg(연속, 기본) | yf(이산, 비교용)
TAG = "" if DIV == "lseg" else "_" + DIV


def label_path(name="mc_variants"):
    """라벨 파일 경로. MCVAR_DIV 한 손잡이로 3스타(mc_variants)와 1·2스타(mc_variants12)를
     **함께** 바꾼다 — 두 파일의 배당 소스가 어긋나면 합친 데이터가 조용히 섞인다."""
    return fm.CACHE / "mcvar" / "{}{}.parquet".format(name, TAG)


def state():
    """지연 적재 — import 만으로 데이터셋을 읽지 않는다.

     라벨 파일은 MCVAR_DIV(배당 소스)로 고르고, MCVAR_FILE 로 직접 지정하면 그쪽이 이긴다
     (둘 다 config.yaml 밖의 설정 통로라 상시 사용은 권장하지 않음)."""
    global _ST
    if _ST is not None:
        return _ST
    cfg0 = yaml.safe_load(open(fm.ROOT / "config.yaml", encoding="utf-8"))
    D0 = MD.load(cfg0)
    V = pd.read_parquet(os.environ.get(
        "MCVAR_FILE", label_path())).set_index("item").loc[D0.ITEM]
    S = pd.read_parquet(fm.source()).set_index("item").loc[D0.ITEM]
    assert np.allclose(S.fair.values, D0.FAIR, atol=1e-6), "공정가 정렬 불일치"
    # 비교 사용 행: MC 입력 정상 & 가격 캐시 신선
    # (stale_px = 기초자산 최신 수익률이 발행일보다 14일 넘게 오래됨 → 30개 제외, 두 변형 동일)
    OK = (V["fail"].isna() & ~V["stale_px"]).values if "stale_px" in V else np.ones(len(V), bool)
    # 폴드 경계(시점)는 정본 그대로 두고 제외 행만 뺀다
    WF = [(tr[OK[tr]], va[OK[va]], te[OK[te]]) for tr, va, te in D0.WF]
    _ST = dict(cfg=cfg0, D0=D0, V=V, S=S, OK=OK, WF=WF, RHO=V[["rho_12", "rho_13", "rho_23"]].values)
    return _ST


def make_D(var):
    """변형 var('A'|'B') 의 stage-1 번들. 정본 D 를 복사해 MC 입력만 갈아끼운다."""
    st = state()
    D = copy.copy(st["D0"])
    V, RHO = st["V"], st["RHO"]
    mc = V[f"mc_{var}"].values.astype("float32")
    if var == "A":
        D.VC = vc_block(V[["sA_1", "sA_2", "sA_3"]].values, RHO)
    else:
        D.VC = vc_block(V[["sB_1", "sB_2", "sB_3"]].values, RHO, V[["q_1", "q_2", "q_3"]].values)
        D.CURVE = V[[f"bu{j}" for j in range(10)]].values.astype("float32")
    D.MC = mc
    rm = np.full(len(mc), np.nan, "float32")       # 제외 행은 recent_margin 풀에도 넣지 않는다
    ok = st["OK"]
    rm[ok] = recent_margin(st["S"].reset_index()[["fair", "isu_ord", "opt_type", "ki_yn"]]
                           .assign(mc=mc)[ok].reset_index(drop=True))
    D.rm, D.WF = rm, st["WF"]
    D.DON = np.concatenate([D.CURVE, D.VC, D.CON], axis=1).astype("float32")
    return D


def oos_sample(per_struct=500, seed=0):
    """구조별 층화 OOS 표본 — 계약조건 민감도의 기준 표본. (D, src, fold, struct, pick)."""
    st = state()
    D = make_D("B")
    src = st["S"].reset_index()
    fold = np.full(D.n, -1, int)
    for k, (tr, va, te) in enumerate(st["WF"]):
        fold[te] = k
    oos = np.where(fold >= 0)[0]
    struct = (src.opt_type.astype(str) + "·"
              + np.where(src.ki_yn == 1, "낙인", "노낙인")).values
    rng = np.random.default_rng(seed)
    pick = [rng.choice(oos[struct[oos] == s], size=min(per_struct, int((struct[oos] == s).sum())),
                       replace=False) for s in np.unique(struct[oos])]
    return D, src, fold, struct, np.sort(np.concatenate(pick))
