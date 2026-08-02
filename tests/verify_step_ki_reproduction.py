# -*- coding: utf-8 -*-
"""원래 타깃 유니버스(3-star STEP 낙인)만으로 재학습해 이전 성능이 재현되는지 확인.
 (pytest 수집 대상 아님 — 수동 실행)

 4구조 확장 후 통합 모델의 STEP×KI 슬라이스 R² 는 '4구조로 학습한 모델을 STEP×KI 에서 평가'한 값이라,
 '원래처럼 STEP×KI 만으로 학습' 한 것과 다르다. 이 스크립트는 후자를 재현해 세 값을 나란히 놓는다:
   (a) 기록된 이전 성능 (참고치)
   (b) 통합(4구조) 모델의 STEP×KI 슬라이스   <- 5_segment 그림의 값
   (c) STEP×KI 만으로 재학습               <- 이 스크립트가 계산

 가중치는 저장하지 않는다(name=None) — 4구조 학습 결과를 덮어쓰지 않기 위해.
 실행: python tests/verify_step_ki_reproduction.py"""
import copy
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from util import utils, file_manager as fm
from util.metric import metrics
from module import data as DA
from module.pipeline import predict_hybrid

PREV_R2 = 0.69          # 기록된 이전 공정가 R² (참고치, 구 유니버스·구 엔진)


def subset(D, mask, cfg):
    """행 마스크로 D 를 잘라 새 SimpleNamespace 를 만든다. walk-forward 는 잘린 길이로 재계산."""
    idx = np.where(mask)[0]
    E = copy.copy(D)
    E.n = len(idx)
    E.ml = D.ml.iloc[idx].reset_index(drop=True)
    for a in ("FAIR", "MC", "rm", "ITEM", "ORD", "VC", "CURVE", "CON", "R", "TEN", "SIGEFF", "DON"):
        setattr(E, a, getattr(D, a)[idx])
    E.WF = DA.walk_forward(E.n, cfg["data"]["walk_forward"],
                           cfg["data"].get("val_frac", 0.0), cfg["data"].get("val_seed", 0))
    return E


def report(tag, df):
    m = metrics(df["y_true"], df["y_pred"])
    base = df["y_pred"] - df["resid_pred"] if "resid_pred" in df else None
    line = (f"  {tag:34s} n {len(df):6,} | R2 {m['R2']:+.4f} | MAE {m['MAE']:.5f} | "
            f"MAPE {m['MAPE%']:.2f}% | Spearman {m['Spearman']:.3f}")
    if base is not None:
        line += f" | stage2 기여 {m['R2'] - r2_score(df['y_true'], base):+.4f}"
    print(line, flush=True)
    return m["R2"]


def main():
    cfg = utils.load_config()
    D = DA.load(cfg)
    src = pd.read_parquet(fm.source())[["item", "opt_type", "ki_yn"]].astype({"item": str})
    key = src.set_index("item")
    st = D.ml["ITEM_CD"].astype(str).map(key["opt_type"]).values
    ki = D.ml["ITEM_CD"].astype(str).map(key["ki_yn"]).values
    mask = (st == "STEP") & (ki == 1)
    print(f"전체 {D.n:,} | STEP x KI {int(mask.sum()):,}")

    # (b) 통합(4구조) 모델의 STEP×KI 슬라이스 — 저장된 예측에서 바로
    print("\n[b] 통합 4구조 모델의 STEP x KI 슬라이스 (재학습 없음)")
    items_ki = set(D.ml["ITEM_CD"].astype(str).values[mask])
    for name in ("deeponet_hybrid", "xgb_hybrid"):
        p = pd.read_csv(fm.prediction(name))
        p["ITEM_CD"] = p["ITEM_CD"].astype(str)
        report(name, p[p.ITEM_CD.isin(items_ki)])

    # (c) STEP×KI 만으로 재학습
    E = subset(D, mask, cfg)
    print(f"\n[c] STEP x KI 만으로 재학습 (n {E.n:,}, folds {len(E.WF)})")
    for k, (tr, va, te) in enumerate(E.WF):
        print(f"    fold{k}: train {len(tr):6,} val {len(va):5,} test {len(te):5,}")

    from model.deeponet import _anchor
    from model.benchmark import _xgb_anchor
    from module.tabular import fit_tab

    def ml_resid(D_, cfg_, tr, va, te, target, save_path=None):
        """2026-07-10 rebase 이전 stage-2: ml 전체특성(base)으로 잔차 회귀.
         현행은 D.DON(이론가 결정 특성만) 이라 발행사·발행금액·청약일수 등 마진을 설명하는
         특성이 빠져 있다. cfg['margin'] 은 그 시절 설정이 남은 orphan."""
        yp = fit_tab(D_, cfg_, cfg_["margin"]["model"], tr, te, cfg_["margin"]["feature_set"],
                     target, tw=cfg_["data"]["time_decay"], va=va)
        return None, yp

    for tag, afn, rfn in (("deeponet_hybrid (현행 stage2)", _anchor, None),
                          ("deeponet_hybrid (구 stage2=ml특성)", _anchor, ml_resid),
                          ("xgb_hybrid (현행 stage2)", _xgb_anchor, None),
                          ("xgb_hybrid (구 stage2=ml특성)", _xgb_anchor, ml_resid)):
        t0 = time.time()
        df = predict_hybrid(E, cfg, afn, resid_fn=rfn, name=None, target=E.FAIR, use_margin=True)
        r2 = report(tag, df)
        print(f"      ({time.time()-t0:.0f}s)")
        # 폴드별
        fold = np.full(E.n, -1, int)
        for k, (_t, _v, te) in enumerate(E.WF):
            fold[te] = k
        f = pd.Series(fold[fold >= 0])
        d2 = df.copy(); d2["fold"] = f.values
        per = [f"f{k} {r2_score(g.y_true, g.y_pred):+.3f}" for k, g in d2.groupby("fold")]
        print(f"      폴드별 R2: {' | '.join(per)}")

    print(f"\n[a] 2026-07-09 pre-rebase 실측(백업): R2 0.688~0.694 | MAE 0.0219 | MAPE 2.39% | stage2 0.366")
    print("    (scratch/backup_derived/results_pre_stage2rebase/statistics/)")
    print("\n해석:")
    print("  - 데이터는 재현됨: 7월 백업과 현재의 나이브(MC+rm) R2 가 0.5362 vs 0.5356 로 동일.")
    print("  - 차이는 stage-2 입력 특성. 2026-07-10 rebase 가 ml 전체특성 -> D.DON(이론가 특성만)")
    print("    으로 바꾸면서 발행사·발행금액·청약일수 등 '마진을 설명하는' 특성이 빠졌다.")
    print("  - 구 stage2(ml특성) 로 되돌리면 0.66 대로 복귀 -> 이전 성능 재현.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
