# -*- coding: utf-8 -*-
"""저장된 model weight 을 로드해 OOS test set 을 직접 forward → 예측 DataFrame 조립.
(2_DeepONet_model 이 예측 CSV 대신 이걸 써서 가중치→예측 재현)
전 모델이 **이론가(MC) 2단계 예측**: y = MC_hat(stage1) + resid(stage2), 마진 없음, y_true = MC.
result/models/<name>_fold{k}.{pt|pkl} 규약은 pipeline 의 저장 규약과 동일."""
import numpy as np
import pandas as pd
import joblib

from util import file_manager as fm
from .train import load_curve_predictor
from .pi_train import load_pi_predictor
from .tabular import load_tab_predictor
from model.stage2 import load_xgb_resid, load_don_resid, load_ml_resid


def _mp(name, k):
    """폴드별 가중치 경로 접두. `pipeline._mp` 와 같은 규약이지만 **디렉터리를 만들지 않는다**.

    여기는 읽기 전용 경로라서다 — 없는 경로면 로더가 FileNotFoundError 를 올려야 한다.

    입력
      name : str — 모델 이름
      k    : int — 폴드 번호
    출력
      str — "result/models/<name>_fold<k>" (확장자 없음)
    """
    return str(fm.RESULT / "models" / f"{name}_fold{k}")


def _load_xgb_anchor(D, path):
    """benchmark._xgb_anchor 가 저장한 .pkl 을 행 인덱스 예측기로 되살린다.

    입력
      D    : Dataset — D.CURVE · D.VC · D.CON 을 가로로 붙여 X 로 만든다
      path : str — .pkl 경로
    출력
      callable(idx) -> ndarray[float32] — MC_hat 예측기
    """
    m = joblib.load(path)["model"]
    X = np.concatenate([D.CURVE, D.VC, D.CON], axis=1)
    return lambda idx: m.predict(X[idx]).astype("float32")


# 직접(단일 단계) 벤치마크: fit_tab .pkl → 이론가(MC) 직접 예측
_DIRECT = ["bench_ridge", "bench_gbm", "bench_lgbm", "bench_xgboost"]

# 이론가(MC) 2단계(하이브리드) 모델 구성: name -> (anchor_type, resid_type)
_HYBRID = {
    "deeponet_hybrid": ("curve", "mltab"),  # 앵커 MSE
    "deeponet_hybrid_l1": ("curve", "mltab"),  # 앵커 L1
    "deeponet_hybrid_mape": ("curve", "mltab"),  # 앵커 MAPE
    "deeponet_hybrid_s2don": ("curve", "margin"),  # stage-2 = DeepONet
    "xgb_hybrid": ("xgb", "mltab"),  # 앵커 XGB
}

# 구조별 라우팅 하이브리드 (model/pi_route.py): 앵커가 폴드당 2개라 위 (anchor,resid) 규약에 안 맞아 분리.
# name -> 재현에 쓸 시드. 다중 시드 결과는 result/statistics/exp_pi_route_seeds.csv 참고.
_ROUTED = {"deeponet_hybrid_pi_lizard": 0}


def _anchor_loader(D, atype, base):
    """앵커 종류에 맞는 로더를 골라 예측기를 만든다 — 확장자도 종류가 정한다.

    입력
      D     : Dataset
      atype : "curve"(.pt, DeepONet) | "pi"(.pt, 상태증강) | "xgb"(.pkl)
      base  : str — 확장자 없는 경로 접두
    출력
      callable(idx) -> ndarray[float32] — MC_hat 예측기
    예외
      ValueError — 모르는 atype
    """
    if atype == "curve":
        return load_curve_predictor(D, base + ".pt")
    if atype == "pi":
        return load_pi_predictor(D, base + ".pt")  # PI 앵커(상태증강) — 발행시점 상태에서 평가
    if atype == "xgb":
        return _load_xgb_anchor(D, base + ".pkl")
    raise ValueError(atype)


def _resid_loader(D, rtype, base):
    """stage-2 종류에 맞는 로더를 골라 예측기를 만든다.

    입력
      D     : Dataset
      rtype : "mltab"(.pkl, ml 전체특성 — 기본) | "tab"(.pkl, XGB on D.DON) |
              "margin"(.pt, DeepONet stage-2)
      base  : str — 확장자 없는 경로 접두
    출력
      callable(idx) -> ndarray — 잔차 예측기
    예외
      ValueError — 모르는 rtype
    """
    if rtype == "mltab":
        return load_ml_resid(D, base + ".pkl")  # 기본 stage2: ml 전체특성 (cfg['margin'])
    if rtype == "tab":
        return load_xgb_resid(D, base + ".pkl")  # XGB on deeponet 블록 D.DON
    if rtype == "margin":
        return load_don_resid(D, base + ".pt")  # DeepONet stage2 (deeponet 블록)
    raise ValueError(rtype)


def predict_all_from_weights(D, cfg):
    """저장 가중치로 전 모델의 공정가(FAIR) OOS 예측을 조립. pipeline 산출 DataFrame 과 동일 스키마.
    직접(단일 단계) 벤치마크: y = 예측(FAIR) (stage 컬럼 없음).
    2단계 하이브리드: y = MC_hat + recent_margin + resid, y_true = FAIR,
                     resid_true = FAIR − MC_hat − recent_margin, mc_true = MC(이론가 참값).

    입력
      D   : Dataset — MC 와 recent_margin 이 **둘 다** 있어야 한다. 두 열을 따로 검사한다
      cfg : dict — config.yaml (현재는 호출 규약을 맞추는 자리다)
    출력
      dict[str -> DataFrame] — 모델 이름에서 OOS 예측 표로.
        직접 4종(stage 열 없음) + 하이브리드 5종(8열) + 라우팅 1종.
        라우팅 모델은 가중치가 없으면 **조용히 건너뛴다**
    예외
      ValueError — mc 또는 recent_margin 이 전부 비었을 때.
        원천 전용 재현에서는 이 경로를 쓰지 않는다
    """
    FAIR, MC, rm = D.FAIR, D.MC, D.rm
    # 공정가 2단계는 세 입력이 다 있어야 한다. **두 열을 따로 검사한다** —
    # module.data.fill_anchor 가 mc 를 정본 라벨로 채우므로 mc 가드만 두면 그 가드는
    # 영영 발동하지 않고, 원천 전용 트리에서 여전히 비어 있는 recent_margin 이
    # y = mc + rm + resid 와 resid_true 로 흘러들어 조용히 NaN 을 퍼뜨린다.
    for nm, v in (("mc", MC), ("recent_margin", rm)):
        if not np.isfinite(np.asarray(v, dtype="float64")).any():
            raise ValueError(
                "{} 참값이 없다 — 공정가 2단계(FAIR = MC_hat + recent_margin + resid) 경로는 "
                "module.mc_shards 산출물이 있어야 한다. 원천 전용 재현에서는 이 경로를 쓰지 "
                "않는다.".format(nm)
            )
    out = {}
    # 직접(단일 단계) 벤치마크
    for name in _DIRECT:
        rows = []
        for k, (tr, va, te) in enumerate(D.WF):
            pred = load_tab_predictor(D, _mp(name, k) + ".pkl")
            yp = np.asarray(pred(te), dtype="float32")
            rows.append(
                pd.DataFrame(
                    {
                        "ITEM_CD": D.ITEM[te],
                        "isu_ord": D.ORD[te],
                        "y_true": FAIR[te],
                        "y_pred": yp,
                    }
                )
            )
        out[name] = pd.concat(rows, ignore_index=True)
    # 2단계 하이브리드
    for name, (atype, rtype) in _HYBRID.items():
        rows = []
        for k, (tr, va, te) in enumerate(D.WF):
            anchor = _anchor_loader(D, atype, _mp(f"{name}_anchor", k))
            resid = _resid_loader(D, rtype, _mp(f"{name}_resid", k))
            mc_te = np.asarray(anchor(te), dtype="float32")
            r_te = np.asarray(resid(te), dtype="float32")
            y = mc_te + rm[te] + r_te
            rows.append(
                pd.DataFrame(
                    {
                        "ITEM_CD": D.ITEM[te],
                        "isu_ord": D.ORD[te],
                        "y_true": FAIR[te],
                        "y_pred": y,
                        "mc_true": MC[te],
                        "mc_pred": mc_te,
                        "resid_true": (FAIR[te] - mc_te - rm[te]).astype("float32"),
                        "resid_pred": r_te,
                    }
                )
            )
        out[name] = pd.concat(rows, ignore_index=True)
    # 구조별 라우팅 하이브리드 (앵커 2개 → 예측 시 계약조건으로 선택). 없으면 조용히 건너뛴다.
    from model.pi_route import load_routed_predictor

    for name, seed in _ROUTED.items():
        if not (fm.RESULT / "models" / f"{name}_seed{seed}_anchor_base_fold0.pt").exists():
            continue
        rows = []
        for k, (tr, va, te) in enumerate(D.WF):
            anchor = load_routed_predictor(D, name, seed, k)
            resid = load_ml_resid(D, _mp(f"{name}_seed{seed}_resid", k) + ".pkl")
            mc_te = np.asarray(anchor(te), dtype="float32")
            r_te = np.asarray(resid(te), dtype="float32")
            rows.append(
                pd.DataFrame(
                    {
                        "ITEM_CD": D.ITEM[te],
                        "isu_ord": D.ORD[te],
                        "y_true": FAIR[te],
                        "y_pred": mc_te + rm[te] + r_te,
                        "mc_true": MC[te],
                        "mc_pred": mc_te,
                        "resid_true": (FAIR[te] - mc_te - rm[te]).astype("float32"),
                        "resid_pred": r_te,
                    }
                )
            )
        out[name] = pd.concat(rows, ignore_index=True)
    return out
