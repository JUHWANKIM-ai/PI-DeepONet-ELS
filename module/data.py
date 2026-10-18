# -*- coding: utf-8 -*-
"""데이터 스키마·생성·로딩. 정통 DeepONet 재설계:
branch = 시장상태(바스켓 vol·corr + 수익률곡선), trunk = 계약(전체 STRK 스케줄 + 배리어 + 쿠폰 + 만기).
PI-DeepONet만 trunk에 평가좌표(S,τ,I)를 추가로 넣고 BS-PDE 물리를 건다.
컬럼명은 raw DART와 직접 비교되도록 raw 이름 활용. 데이터셋은 CSV.
"""
import bisect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from util import file_manager as fm
from .schedule import NSTRK, SCHED_COLS, LZ_BARR_NEUTRAL, LZ_PMT_NEUTRAL

# ===== 원천 컬럼명 -> 데이터셋 컬럼명 (raw 이름/산식 활용) =====
RENAME = {
    "fair": "FAIR_VALUE/ISU_PRC_DETAIL",
    "coupon": "ANL_RTRN/100",
    "B": "BARR_1/100",
    "tenor": "TENOR",
    "isu_ord": "ISU_DT_ordinal",
    "mc": "MC",
    # pass-through
    "amt": "ACT_ISU_AMT",
    "sbrt": "SB_RT",
    "dvrt": "DV_RT",
    "prcp": "PRCP_GRTE_RT",
    "kigrc": "KNCK_IN_GRC_PRD",
    "issuer": "ISU_ORG",
    "risk": "RISK_GRADE",
    "ptype": "PRODUCT_TYPE",
    "rdmp": "RDMP_TYPE",
    "iyear": "ISU_DT_year",
    "imonth": "ISU_DT_month",
    "subdays": "SUB_END_DT-SUB_START_DT",
    "K": "STRK_1_last/100",
    "Kfirst": "STRK_1_first/100",
    "opt_type": "OPT_TYPE",
    "ki_yn": "KNCK_IN_YN",  # 구조 (STEP/LIZARD × 낙인/노낙인)
    "item": "ITEM_CD",  # 행 식별 인덱스 (훈련 미사용, 예측→상품 추적용)
}
INV = {v: k for k, v in RENAME.items()}


def _new(name):
    """원천 컬럼명을 데이터셋 컬럼명으로 바꾼다 (RENAME 에 없으면 그대로).

    입력
      name : str — 원천 이름 (예: "fair")
    출력
      str — 데이터셋 이름 (예: "FAIR_VALUE/ISU_PRC_DETAIL").
            역방향은 모듈 상수 INV 다
    """
    return RENAME.get(name, name)


# ===== 피처 그룹 =====
# tabular (ml.csv): 벤치마크 + 하이브리드 stage-2 마진모델
BASE = [
    _new(c)
    for c in [
        "sig1",
        "sig2",
        "sig3",
        "rho12",
        "rho13",
        "rho23",
        "sig_mean",
        "rho",
        "r",
        "B",
        "K",
        "Kfirst",
        "coupon",
        "tenor",
        "nobs",
        "cpn_spread",
        "b_over_k",
        "stepdown",
        "mom6m",
        "amt",
        "sbrt",
        "dvrt",
        "prcp",
        "kigrc",
        "iyear",
        "subdays",
    ]
]
REG = [
    "recent_margin",
    "recent_mktvol",
    "curve_level",
    "curve_slope",
    "curve_curv",
    "issue_intensity",
]
CAT = [_new(c) for c in ["issuer", "risk", "ptype", "rdmp", "imonth"]]
# 인과(as-of, 미래정보 無) stage-2 드라이버: 발행사 마진편차(shrinkage)·신뢰도·IV−HV 스프레드·IV수준.
# 오차분석의 발행사 이질성·위험프리미엄을 명시적으로 모델링 (절대수준 regime 피처는 OOS 실패 → 상대신호만).
# 균등 m_issuer 는 m_issuer_ewma 가 흡수+노이즈라 제거(ablation: 빼면 +0.004). 5개 세트가 OOS 최적.
DRIVER = ["m_issuer_ewma", "m_issuer_cnt", "iv_hv_spread", "sig_iv_lr", "recent_mktvol"]

# 연산자망 (deeponet.csv)
VOLCORR = [
    "sig1",
    "sig2",
    "sig3",
    "rho12",
    "rho13",
    "rho23",
    "sig_eff",
]  # branch: 바스켓 변동성·상관
UC = [f"u{j}" for j in range(10)]  # branch: 수익률곡선 10노드
STRK = [f"strk_{j}" for j in range(NSTRK)]  # trunk: 오토콜 행사가 스케줄(/100, 패딩)
PMT = [f"pmt_{j}" for j in range(NSTRK)]  # trunk: 회차별 누적 지급률(소수)
LZB = [f"lz_barr_{j}" for j in range(NSTRK)]  # trunk: 리자드 배리어(없으면 중립 1.2)
LZP = [f"lz_pmt_{j}" for j in range(NSTRK)]  # trunk: 리자드 축소 지급률(없으면 0)

# ml BASE 에도 행사가·지급률 스케줄 포함. K/Kfirst 는 파생피처(b_over_k, stepdown)용으로 유지.
# 리자드 배열은 ml 에 넣지 않고 CAT 의 OPT_TYPE 으로 구조를 구분한다(GBM 은 범주형이 더 유리).
BASE = BASE + STRK + PMT
CONTRACT = STRK + PMT + LZB + LZP + [_new("B"), _new("coupon"), _new("tenor")]  # trunk 51
CAT = CAT + [_new("opt_type"), _new("ki_yn")]

INDEX = _new("item")  # ITEM_CD (행 식별 인덱스, 훈련 미사용)
TARGET = _new("fair")  # FAIR_VALUE/ISU_PRC_DETAIL
ANCHOR = _new("mc")  # MC
MARGIN = "recent_margin"
ORDER = _new("isu_ord")  # ISU_DT_ordinal
SRC_ORDER = "isu_ord"
TENOR = _new("tenor")  # TENOR
BARR = _new("B")  # BARR_1/100
COUPON = _new("coupon")  # ANL_RTRN/100
RF = "r"  # 무위험금리 (물리용)
SIGEFF = "sig_eff"
CSV_ENC = "utf-8-sig"


def featnum(feat):
    """피처셋 이름으로 수치 특성 열 목록을 만든다.

    입력
      feat : "base" | "regime" | "driver" — regime 은 BASE+REG, driver 는 BASE+DRIVER.
             그 밖의 값은 **조용히 BASE 만** 준다 (예외를 올리지 않는다)
    출력
      list[str] — 열 이름 목록. 범주형(CAT)은 포함하지 않는다
    """
    return BASE + (REG if feat == "regime" else []) + (DRIVER if feat == "driver" else [])


def fill_lizard_neutral(df):
    """리자드 없는 회차의 NaN 을 중립값으로 채운다 (parquet 은 NaN 유지, CSV·모델 입력만 채움).

    lz_barr=1.2 는 스팟 위라 no-touch 조건이 즉시 깨져 '발동 불가' 가 명확하다.

    입력
      df : DataFrame — lz_barr_* · lz_pmt_* 열. **제자리에서 수정한다**
    출력
      DataFrame — 같은 객체. 해당 열이 없으면 건너뛴다
    """
    for j in range(NSTRK):
        b, p = f"lz_barr_{j}", f"lz_pmt_{j}"
        if b in df.columns:
            df[b] = df[b].fillna(LZ_BARR_NEUTRAL)
        if p in df.columns:
            df[p] = df[p].fillna(LZ_PMT_NEUTRAL)
    return df


def _load_source():
    """원천 parquet 을 읽고 발행 밀집도(issue_intensity)를 붙인다.

    행사가·지급률·리자드 스케줄이 이미 parquet 에 있으므로 raw 를 다시 읽지 않는다.

    입력
      없음 (data/els3_dataset.parquet)
    출력
      DataFrame — 발행일 오름차순. issue_intensity(발행 전 90일 발행 건수) 추가,
                  리자드 중립값 채움
    예외
      AssertionError — 스케줄 48열이 parquet 에 없을 때 (0_data 재실행이 필요하다)
    """
    df = pd.read_parquet(fm.source()).sort_values(SRC_ORDER).reset_index(drop=True)
    o = df[SRC_ORDER].tolist()
    df["issue_intensity"] = [
        bisect.bisect_left(o, o[i]) - bisect.bisect_left(o, o[i] - 90) for i in range(len(df))
    ]
    missing = [c for c in SCHED_COLS if c not in df.columns]
    assert (
        not missing
    ), f"parquet 에 스케줄 컬럼 없음 (0_data build_source 재실행 필요): {missing[:4]}"
    return fill_lizard_neutral(df)


# ===== 0_data: 데이터셋 생성 (CSV) =====
def _label_path():
    """정본 MC 라벨 파일. module.mcvar 가 MCVAR_DIV 로 고르는 것과 같은 경로.

    module.mcvar 는 module.data 를 쓰므로 모듈 최상단에서 불러올 수 없다 — 함수 안에서 늦게 부른다.

    입력
      없음
    출력
      Path — data/cache/mcvar/mc_variants<TAG>.parquet. 존재 여부는 확인하지 않는다
    """
    from . import mcvar as MV

    return MV.label_path()


STAMP_FILE = ".dataset_label_stamp.json"  # 데이터셋이 어떤 라벨로 채워졌는지


def dataset_stamp():
    """데이터셋이 채워진 라벨의 출처를 읽는다.

    입력
      없음 (data/.dataset_label_stamp.json)
    출력
      dict | None — 라벨 스탬프. 파일이 없거나 깨져 있으면 None
                    (읽기 실패도 None 으로 본다 — 출처를 증명할 수 없으면 모르는 것이다)
    """
    import json

    p = fm.DATA / STAMP_FILE
    if not Path(str(p)).exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def record_dataset_stamp(st):
    """데이터셋을 채운 라벨의 출처를 남긴다.

    라벨에 스탬프가 없으면(스탬프 도입 이전 파일) `{"unstamped": true}` 를 남긴다 —
    "출처를 모르는 라벨로 채웠다" 는 것도 정보다.

    입력
      st : dict | None — 라벨 스탬프. None 이면 {"unstamped": True} 를 쓴다
    출력
      없음 (data/.dataset_label_stamp.json 을 덮어쓴다)
    """
    import json

    body = {"unstamped": True} if st is None else st
    (fm.DATA / STAMP_FILE).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")


def datasets_stale():
    """지금 라벨로 데이터셋을 다시 써야 하는가.

    종전 조건은 "MC 가 전부 NaN" 이었다. 그러면 라벨이 **바뀐** 경우는 걸리지 않고
    낡은 라벨이 조용히 배포된다 — 이 브랜치가 다른 곳에서 고치고 있는 바로 그 결함이다.

    ⚠ 알려진 계층 역전: `module` 이 `generate` 를 import 한다. 스탬프 비교 규칙
    (`stamp_differs`)의 정본이 그쪽에 있어서인데, 방향은 반대가 맞다 — 별도 과제다.

    입력
      없음
    출력
      bool — True 면 데이터셋을 다시 써야 한다.
             라벨 파일 자체가 없으면 **False**(채울 것이 없으므로 판단 보류)
    """
    from . import mcvar  # noqa: F401  (경로 해석용 늦은 import)
    import generate.build_mc_variants as BMV

    want = BMV.read_stamp(_label_path())
    if want is None:
        return False  # 라벨 자체가 없다 — 채울 것이 없으므로 판단 보류
    return BMV.stamp_differs(dataset_stamp(), want)


def fill_anchor(df):
    """`mc` 열의 빈 자리를 정본 라벨(`mc_B`)로 채우고 채운 행 수를 돌려준다.

    Task 4 에서 `mc` 열 의존을 끊어 열만 만들고 NaN 으로 뒀는데, 그 결과 README 가 데이터셋으로
    내세우는 `ml.csv`·`deeponet.csv` 의 라벨 열이 통째로 비었다. 빈 라벨 열은 조용히 오용된다.

    이미 값이 있는 행은 건드리지 않는다 (공정가 시대 `mc` 를 덮어쓰면 안 된다).
    라벨 파일이 없으면 0 을 돌려주고 아무것도 하지 않는다 — 차가운 트리에서는 노트북 0 이
    노트북 1 보다 먼저 돌기 때문이다. 노트북 1 이 라벨을 만든 뒤 데이터셋을 다시 쓴다.

    입력
      df : DataFrame — `item` 과 원천 `mc` 열. **제자리에서 수정한다**
    출력
      int — 채운 행 수. 라벨 파일이 없으면 0
    """
    src = INV.get(ANCHOR, ANCHOR)
    try:
        lab = _label_path()
    except Exception:
        return 0
    if not Path(str(lab)).exists():
        return 0
    L = pd.read_parquet(lab, columns=["item", "mc_B"]).set_index("item")["mc_B"]
    need = df[src].isna() if src in df.columns else pd.Series(True, index=df.index)
    got = df["item"].map(L)
    hit = need & got.notna()
    df.loc[hit, src] = got[hit].values
    return int(hit.sum())


def build_datasets():
    """원천 -> data/ml.csv, data/deeponet.csv 생성.
    ml = tabular(BASE+REG+CAT), deeponet = 연산자망 입력(곡선 u0-9 + vol·corr·sig_eff + 계약 + r).
    둘 다 공통(fair·mc·recent_margin) 포함.

    공정가 시대 열(mc · recent_margin)은 `module.mc_shards.combine` 이 붙인다. 원천 전용
    재현에서는 그 샤드 캐시가 없으므로 **열만 만들고 NaN 으로 둔다** — 스키마가 깨지면 안 되고,
    D.MC 는 어차피 `module.mcvar.make_D` 가 변형 B 로 갈아끼운다.

    입력
      없음 (원천 parquet · 라벨 parquet)
    출력
      dict — {"ml": 행수, "deeponet": 행수, "columns": {"ml": [...], "deeponet": [...]}}.
             CSV 두 개를 data/ 에 쓰고, 라벨로 채웠으면 스탬프도 남긴다
    """
    fm.ensure_dirs()
    df = _load_source()
    common = [
        INDEX,
        ORDER,
        TARGET,
        ANCHOR,
        MARGIN,
    ]  # INDEX(ITEM_CD) 선두 — 모든 데이터셋에 인덱스 컬럼
    # 공정가 시대 열(mc · recent_margin)은 module.mc_shards.combine 이 붙인다. 원천 전용
    # 재현에서는 그 샤드 캐시가 없으므로 **열만 만들고 NaN 으로 둔다** — module.data.load 의
    # 스키마가 깨지면 안 되고, D.MC 는 어차피 module.mcvar.make_D 가 변형 B 로 갈아끼운다.
    for c in (ANCHOR, MARGIN):
        src_col = INV.get(c, c)
        if src_col not in df.columns:
            df[src_col] = np.nan
    if fill_anchor(df):  # 라벨 파일이 있으면 MC 열을 채운다 (없으면 NaN 으로 남는다)
        import generate.build_mc_variants as _BMV

        record_dataset_stamp(_BMV.read_stamp(_label_path()))  # 어떤 라벨로 채웠는지 남긴다

    def write(name, new_cols):
        """데이터셋 하나를 CSV 로 쓴다 — 열 중복은 **순서를 지키며** 제거한다.

        입력
          name     : "ml" | "deeponet"
          new_cols : list[str] — 데이터셋 이름 기준 열 목록 (BASE 가 STRK·PMT 와 겹친다)
        출력
          list[str] — 실제로 쓴 열 이름
        """
        new_cols = list(dict.fromkeys(new_cols))
        old_cols = [INV.get(c, c) for c in new_cols]
        out = df[old_cols].rename(columns=RENAME)
        out.to_csv(fm.dataset(name), index=False, encoding=CSV_ENC)
        return list(out.columns)

    drv = [
        c for c in DRIVER if c in df.columns
    ]  # 드라이버는 존재할 때만 포함(inject_drivers 선행 시)
    cols_ml = write("ml", [INDEX] + BASE + REG + drv + CAT + common)
    # DeepONet 연산자망 입력: 곡선 u0-9 + vol·corr·sig_eff + 계약(strk0-11,B,coupon,tenor) + r (stage1 앵커/stage2 D.DON 공용)
    cols_dn = write("deeponet", [INDEX] + UC + VOLCORR + CONTRACT + [RF] + common)
    return {"ml": len(df), "deeponet": len(df), "columns": {"ml": cols_ml, "deeponet": cols_dn}}


# ===== 1_run: 통합 로딩 =====
def load(cfg):
    """두 CSV 를 읽어 학습·추론이 공유하는 번들 D 를 만든다 — **정본 D 의 정의**다.

    두 파일을 같은 키(ORDER)로 정렬해 행을 맞춘다. 행 수가 다르면 거부한다.

    입력
      cfg : dict — config.yaml. data.smax · data.walk_forward · data.val_frac ·
            data.val_seed · device 를 읽는다
    출력
      SimpleNamespace — n · SMAX · DEV · ml(표) ·
        FAIR · MC · rm · ITEM · ORD —— 라벨·식별자
        VC(n,7) · CURVE(n,10) · CON(n,51) · R · TEN · SIGEFF —— 연산자망 입력
        DON(n,68) —— stage-2 입력 = [CURVE | VC | CON]
        iK · iKlast · iBARR · iCOUPON · iTEN —— CON 안의 열 인덱스 (물리·페이오프용)
        WF —— walk-forward 폴드
      ⚠ `module.mcvar.make_D` 가 돌려주는 변형 D 는 VC 가 10열이라 **다른 타입**이다
    예외
      AssertionError — 두 CSV 의 행 수가 다를 때
    """
    ml = pd.read_csv(fm.dataset("ml"), encoding=CSV_ENC).sort_values(ORDER).reset_index(drop=True)
    don = (
        pd.read_csv(fm.dataset("deeponet"), encoding=CSV_ENC)
        .sort_values(ORDER)
        .reset_index(drop=True)
    )
    n = len(ml)
    assert len(don) == n, "데이터셋 행수 불일치"
    for c in CAT:
        ml[c] = ml[c].astype(str).astype("category")

    dev = "cuda" if (cfg.get("device") != "cpu" and torch.cuda.is_available()) else "cpu"
    D = SimpleNamespace()
    D.n = n
    D.SMAX = float(cfg["data"]["smax"])
    D.DEV = dev
    D.ml = ml
    D.FAIR = ml[TARGET].values.astype("float32")
    D.MC = ml[ANCHOR].values.astype("float32")
    D.rm = ml[MARGIN].values.astype("float32")
    D.ITEM = ml[INDEX].astype(str).values  # ITEM_CD 인덱스 (훈련 미사용, 예측 추적용)
    D.ORD = ml[ORDER].values.astype(float)
    # 연산자망 입력 (branch=vol·corr+곡선, trunk=계약) — deeponet.csv
    D.VC = don[VOLCORR].values.astype("float32")  # (n, 7)
    D.CURVE = don[UC].values.astype("float32")  # (n, 10)
    D.CON = don[CONTRACT].values.astype(
        "float32"
    )  # (n, 51) = strk/pmt/lz_barr/lz_pmt ×12 + BARR + coupon + TENOR
    D.R = don[RF].values.astype("float32")  # r
    D.TEN = don[TENOR].values.astype("float32")  # 만기
    D.SIGEFF = don[SIGEFF].values.astype("float32")  # σ_eff
    # stage-2 잔차모델 입력 = deeponet.csv 특성 블록 [곡선|vol·corr·sig_eff|계약] (= 앵커 입력, 이론가 결정 특성만; ml aux/범주형 제외)
    D.DON = np.concatenate([D.CURVE, D.VC, D.CON], axis=1).astype("float32")  # (n, 68) = 10+7+51
    # CONTRACT 내 인덱스 (물리/payoff용)
    D.iK = CONTRACT.index(f"strk_{NSTRK - 1}")  # 만기 행사가 = strk_{last}
    D.iKlast = CONTRACT.index(f"strk_{NSTRK - 1}")
    D.iBARR = CONTRACT.index(BARR)
    D.iCOUPON = CONTRACT.index(COUPON)
    D.iTEN = CONTRACT.index(TENOR)
    D.WF = walk_forward(
        n,
        cfg["data"]["walk_forward"],
        cfg["data"].get("val_frac", 0.0),
        cfg["data"].get("val_seed", 0),
    )
    return D


def walk_forward(n, bounds, val_frac=0.0, val_seed=0):
    """확장윈도우 walk-forward 폴드를 만든다.

    va 는 train 에서 **무작위로** 뽑으므로 미래 참조를 허용한다 — 조기종료 전용이고,
    성능은 te 로만 잰다.

    입력
      n        : int — 전체 행 수. 경계는 비율이라 여기에 곱해 인덱스로 쓴다
      bounds   : list[float] — 누적 비율 경계. 인접 쌍이 한 폴드다 (경계 k개 → 폴드 k−1개)
      val_frac : float — train 에서 va 로 뺄 비율. 0 이면 va 는 빈 배열
      val_seed : int — va 추출 시드
    출력
      list[(tr, va, te)] — 각각 ndarray[int].
        tr = 과거 전체에서 va 를 뺀 것, te = 그 다음 구간 (미래 OOS)
    """
    rng = np.random.default_rng(val_seed)
    folds = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tr_all = np.arange(0, int(a * n))
        te = np.arange(int(a * n), int(b * n))
        if val_frac and len(tr_all) > 10:
            nv = max(1, int(round(val_frac * len(tr_all))))
            va = np.sort(rng.choice(tr_all, size=nv, replace=False))
            tr = np.setdiff1d(tr_all, va, assume_unique=True)
        else:
            va = np.empty(0, dtype=int)
            tr = tr_all
        folds.append((tr, va, te))
    return folds


def time_weights(D, tr, half_life=365.25):
    """시간감쇠 표본가중 0.5^(dt/half_life).

    기준점은 **학습 폴드의 최신일**이다 — 전체 최신일이 아니므로 폴드마다 기준이 다르다.

    입력
      D         : Dataset — D.ORD(발행일 ordinal)를 본다
      tr        : ndarray[int] — 학습 행
      half_life : float | None — 반감기 [일]. None 이면 균등가중(전부 1)
    출력
      ndarray(len(tr),) float32 — 최신 행이 1.0, 과거로 갈수록 감쇠
    """
    if half_life is None:
        return np.ones(len(tr), dtype="float32")
    return (0.5 ** ((D.ORD[tr].max() - D.ORD[tr]) / half_life)).astype("float32")


def to_tensor(a, dev):
    """배열을 float32 torch 텐서로 올린다.

    입력
      a   : array-like — numpy 배열 또는 그로 바꿀 수 있는 것
      dev : str — torch 장치
    출력
      Tensor — dtype float32. 원본과 메모리를 공유하지 않는다
    """
    return torch.tensor(np.asarray(a, dtype="float32"), device=dev)


ZCLIP = 10.0  # 표준화값 한계 (정규 표본이면 절대 안 걸리는 수준)


def zstats(a, tr):
    """train 기준 표준화 통계. **학습 폴드만** 보므로 누출이 없다.

    입력
      a  : ndarray(n, k) — 표준화할 행렬
      tr : ndarray[int] — 학습 행
    출력
      (m, s) — 각각 ndarray(k,). s 에는 1e-8 을 더해 0 나눗셈을 막는다
               (그 eps 때문에 상수 열이 폭주하는 문제는 `znorm` 이 클리핑으로 막는다)
    """
    return a[tr].mean(0), a[tr].std(0) + 1e-8


def znorm(a, m, s, clip=ZCLIP):
    """train 기준 표준화 + 클리핑.

    train 폴드에서 상수인 열(예: 리자드 lz_pmt_5/6/7 — 그 회차에 리자드가 붙은 상품이
    train 구간에 하나도 없다)은 std 가 eps(1e-8) 라, test 에 값이 나타나면 |z| 가
    2,500만까지 튄다. 트리 모델은 스케일 불변이라 무사하지만 tanh MLP 는 포화돼 발산한다.
    (실측: deeponet_hybrid_s2don stage2 R² −1.84)

    입력
      a    : ndarray(n, k) — 표준화할 행렬
      m, s : ndarray(k,) — `zstats` 의 출력
      clip : float — 절대값 상한. 기본 ZCLIP(10.0) — 정규 표본이면 걸리지 않는 수준이다
    출력
      ndarray(n, k) — (a − m)/s 를 ±clip 으로 자른 것
    """
    return np.clip((a - m) / s, -clip, clip)
