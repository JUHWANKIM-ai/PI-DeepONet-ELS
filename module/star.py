# -*- coding: utf-8 -*-
"""1·2·3스타(기초자산 1~3개)를 합친 stage-1 데이터 묶음 — 같은 입력 규약 · 같은 walk-forward.

 3스타: module.mcvar.make_D("B") 가 주는 배열 그대로.
 1·2스타: els12_dataset + mcvar/mc_variants12 로 같은 배열을 만든다 (슬롯 중복 패딩:
          1스타는 A,A,A / 2스타는 A,B,B 로 채워 3자산 규약에 맞춘다. 중복쌍 상관은 정확히 1.0).
 합친 뒤 발행일로 정렬하고 walk-forward 폴드를 다시 끊는다 (config 의 경계 그대로).

 반환 D 는 CURVE/VC/CON/MC/FAIR/ORD/ITEM/nstar/nobs/WF/n/DEV 를 가진 네임스페이스다.
 ⚠ 정본 `module.data.load` 의 D 와 **다른 타입**이다 — R/TEN/SIGEFF/rm/ml/DON/SMAX 와
 iK/iBARR/iCOUPON/iTEN 이 없고, 대신 nstar/nobs 가 있다. 같은 이름의 다른 타입이라는
 스키마 문제는 module/mcvar.py 머리말에 적어둔 별도 과제다.

 els12_dataset.parquet 은 tools/build_12star.py 가, mc_variants12 는 tools/build_mc_variants.py 가 만든다.
 종전에는 scratch/stage1_all_stars.py 에만 있어 추적 파이프라인에서 쓸 수 없었다.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from util import file_manager as fm, utils
from . import mcvar as MV
from .data import CONTRACT, walk_forward
from .schedule import LZ_BARR_NEUTRAL, LZ_PMT_NEUTRAL
from .vbase import vbase_els

CACHE_VB = fm.CACHE / "els_vbase_clark_all.npy"


def _con12(d):
    """els12_dataset -> CONTRACT(51) 배열. 이름이 다른 3열 매핑 + 리자드 중립값 채우기.

     리자드가 없는 상품은 원천에 lz_barr/lz_pmt 가 비어 있다. 3스타 deeponet.csv 와 같은 규약으로
     **중립값**(배리어 1.2 = 절대 발동 안 함, 지급률 0)으로 채운다.
    """
    ren = {"BARR_1/100": "B", "ANL_RTRN/100": "coupon", "TENOR": "tenor"}
    X = d[[ren.get(c, c) for c in CONTRACT]].copy()
    for c in CONTRACT:
        src = ren.get(c, c)
        if c.startswith("lz_barr_"):
            X[src] = X[src].fillna(LZ_BARR_NEUTRAL)
        elif c.startswith("lz_pmt_"):
            X[src] = X[src].fillna(LZ_PMT_NEUTRAL)
    return X.values.astype("float32")


def make_all(var="B"):
    D3 = MV.make_D(var)
    ok3 = MV.state()["OK"]
    d12 = pd.read_parquet(fm.DATA / "els12_dataset.parquet").sort_values("isu_ord").reset_index(drop=True)
    V12 = pd.read_parquet(fm.CACHE / "mcvar" / "mc_variants12.parquet")
    V12 = V12.set_index("item").reindex(d12.item.values)
    ok12 = V12["mc_B"].notna().values & (V12.get("fail", pd.Series(index=V12.index, dtype=object)).isna().values)
    ok12 &= np.isfinite(V12[["sB_1", "sB_2", "sB_3"]].values).all(1)
    ok12 &= np.isfinite(V12[["bu{}".format(j) for j in range(10)]].values).all(1)
    d12, V12 = d12[ok12].reset_index(drop=True), V12[ok12].reset_index(drop=True)

    VC12 = MV.vc_block(V12[["sB_1", "sB_2", "sB_3"]].values,
                      V12[["rho_12", "rho_13", "rho_23"]].values,
                      V12[["q_1", "q_2", "q_3"]].values)
    CU12 = V12[["bu{}".format(j) for j in range(10)]].values.astype("float32")
    CON12 = _con12(d12)
    MC12 = V12["mc_B"].values.astype("float32")

    i3 = np.where(ok3)[0]
    parts = dict(
        CURVE=np.concatenate([D3.CURVE[i3], CU12]),
        VC=np.concatenate([D3.VC[i3], VC12]),
        CON=np.concatenate([D3.CON[i3], CON12]),
        MC=np.concatenate([D3.MC[i3], MC12]),
        FAIR=np.concatenate([D3.FAIR[i3], d12["fair"].values.astype("float32")]),
        ORD=np.concatenate([D3.ORD[i3], d12["isu_ord"].values.astype(float)]),
        ITEM=np.concatenate([D3.ITEM[i3], d12["item"].astype(str).values]),
        nstar=np.concatenate([np.full(len(i3), 3), d12["nstar"].values.astype(int)]),
        nobs=np.concatenate([MV.state()["S"].reset_index().nobs.values[i3].astype(int),
                             d12["nobs"].values.astype(int)]))
    o = np.argsort(parts["ORD"], kind="stable")
    D = SimpleNamespace(DEV=D3.DEV)
    for k, v in parts.items():
        setattr(D, k, v[o])
    D.n = len(D.MC)
    cfg = utils.load_config()
    D.WF = walk_forward(D.n, cfg["data"]["walk_forward"],
                        cfg["data"].get("val_frac", 0.0), cfg["data"].get("val_seed", 0))
    return D


def vbase_all(D, cache=True):
    if cache and CACHE_VB.exists():
        vb = np.load(CACHE_VB)
        if len(vb) == D.n:
            return vb
    vb = vbase_els(D.CON, D.VC, D.CURVE, D.nobs, mode="clark", nstar=D.nstar)
    vb = np.nan_to_num(vb.astype("float32"), nan=0.0)
    if cache:
        np.save(CACHE_VB, vb)
    return vb
