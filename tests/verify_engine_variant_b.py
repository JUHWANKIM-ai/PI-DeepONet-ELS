# -*- coding: utf-8 -*-
"""추적 패키지만으로 '현행 방법론'(변형 B)이 재현되는지 — pytest 수집 대상 아님, 수동 실행.

 현행 방법론 = 6개월 EWMA 변동성(λ=0.99) + KRX 영업일 120일 쌍별 상관
               + CD 91일·IRS par 부트스트랩 제로커브 + **연속 배당수익률 q** + 일별 GBM.

 기준은 저장된 `data/cache/mcvar/mc_variants.parquet` 의 `mc_B` 열이다
 (경로 수는 `tools.build_mc_variants.NPATH`, 공통난수 seed = mc_seed).
 같은 입력·같은 시드로 다시 매기면 비트 단위로 같아야 한다.

 **경로 수와 배당 소스를 상수로 적지 않는다.** 라벨을 만든 모듈에서 가져온다 —
 종전에는 40,000 과 `st["divlog"]` 를 적어 넣었고, 라벨이 10만 경로·연속 q 로 바뀐 뒤
 (배당 0 으로 매긴 가격을 연속 q 라벨과 대조해) 항상 불일치를 보고하게 됐다.

 종전에는 scratch/mc_discrete_div.py 의 엔진 복사본을 기준으로 대조했다. 그 포크를 없애고
 module/{features,curve,dividend,mc_engine} 로 올린 뒤에는 저장 라벨이 유일한 외부 기준이다.

 합격 기준: 액면 1.0 기준 1e-9 이내.
 실행: python tests/verify_engine_variant_b.py [상품수]
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOL = 1e-9
NPROD = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def main():
    from module import market as MK, mc_engine as ME
    from tools import build_mc_variants as BMV      # 라벨의 경로 수·배당 소스의 출처
    from module.schedule import LZ_BARR_NEUTRAL
    from util import file_manager as fm

    df = MK.dataset()
    V = pd.read_parquet(fm.CACHE / "mcvar" / "mc_variants.parquet").set_index("item")
    RET, KBD = MK.returns(), None
    KBD = MK.calendar(RET)

    rng = np.random.default_rng(0)
    gaps, done, skipped = [], 0, 0
    for i in rng.choice(len(df), min(NPROD * 3, len(df)), replace=False):
        if done >= NPROD:
            break
        row = df.iloc[int(i)]
        if row["item"] not in V.index or not np.isfinite(V.loc[row["item"], "mc_B"]):
            skipped += 1
            continue
        st = MK.state_for(row, ret=RET, cal=KBD)
        if st is None:
            skipped += 1
            continue
        n = int(row["nobs"])
        got = float(ME.mc_daily_t(
            st["sig"], st["corr"], st["dz"], float(row["B"]),
            [float(row[f"strk_{j}"]) for j in range(n)], float(row["tenor"]),
            n=BMV.NPATH, seed=int(row["mc_seed"]), c=float(row["coupon"]),
            pmts=[float(row[f"pmt_{j}"]) for j in range(n)],
            lz_barr=[float(row[f"lz_barr_{j}"]) if row[f"lz_barr_{j}"] < LZ_BARR_NEUTRAL
                     else np.nan for j in range(n)],
            lz_pmt=[float(row[f"lz_pmt_{j}"]) for j in range(n)],
            **st["dkw"]))
        ref = float(V.loc[row["item"], "mc_B"])
        gaps.append(abs(got - ref)); done += 1
        print("  {:22} 저장 {:.8f} / 재계산 {:.8f} | 차이 {:.2e}".format(
            str(row["item"])[:22], ref, got, gaps[-1]))

    if not done:
        print("대조 가능한 상품이 없었다 — mcvar 캐시와 가격 캐시를 확인할 것")
        return 1
    worst = max(gaps)
    print()
    print("대조 {}건 (건너뜀 {}) | 최대 차이 {:.3e} (허용 {:.0e})".format(done, skipped, worst, TOL))
    print("결과:", "일치 — 추적 패키지만으로 현행 방법론이 재현된다" if worst <= TOL
          else "불일치 — 입력 정의를 다시 확인할 것")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    sys.exit(main())
