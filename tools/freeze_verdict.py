# -*- coding: utf-8 -*-
"""재현 판정의 근거를 저장소 안에 고정한다 — 기준선 스냅샷을 지워도 감사 가능하도록.

 `scratch/raw_run_baseline/` 는 git-ignore 된 되돌림용 사본이고 사람 승인 후 지운다.
 판정의 입력이 거기에만 있으면 지우는 순간 결론만 남고 근거가 사라진다.

 담는 것 (데이터가 아니라 **판정 근거**다):
   px_effect      공통난수 가격계열 측정 200건의 부분집합별 요약 (n · 평균|차이| · 편향 · SE · 상한)
   model_old      구본 모델 성능 ALL·통합학습 한 행 (지표별 값과 SE)
   label_summary  구본 라벨과의 라벨차 요약 (부분집합별 n_cmp · mad · se · bias · corr)

 전량 라벨은 담지 않는다 — 그건 데이터이고, 푸시 금지 대상이다.

 실행: python tools/freeze_verdict.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from util import file_manager as fm              # noqa: E402
import tools.verify_raw_run as V                  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "raw_run_verdict.json"
NOTES = (
    "판정량은 px_mad (공통난수로 MC 잡음이 상쇄된 가격계열 단독 효과). "
    "mc_mad 는 구본·신본이 공통난수가 아니라(mc_seed 가 행 번호라 123건 유입으로 98.2%가 바뀜) "
    "판정에 쓸 수 없다. 기준선 5/10원은 계획서가 mc_mad 에 대해 유도한 값이므로 "
    "계통 편차인 px_mad 에 그대로 옮겨오지 않는다 — IDX 는 그래서 미결론이다."
)


def build():
    # 기준선 스냅샷이 없으면 만들 수 없다. 픽스처는 **그 시점의 기록**이고 지금 자료로 다시
    # 만들 수 있는 것이 아니다 — model_old 는 구본 성능이고 구본은 이미 지워졌다.
    # 승인 후 삭제가 정상 경로이므로, 불명확한 TypeError 대신 그 사실을 말한다.
    if not (BASE / "statistics" / V.IMPACT).exists():
        raise SystemExit(
            "기준선 스냅샷이 없다 ({}).\n"
            "판정 근거는 이미 {} 에 고정돼 있고, 그것이 유일한 기록이다.\n"
            "다시 만들려면 스냅샷이 필요하다 — 지금 자료로는 재구성할 수 없다.".format(
                BASE, OUT.relative_to(ROOT)))
    px = {s: dict(n=int(n), mad=float(mad), bias=float(bias),
                  se=None if not np.isfinite(se) else float(se),
                  hi=None if not np.isfinite(hi) else float(hi))
          for s, (n, mad, bias, se, hi) in V.px_effect().items()}

    ro = V._all_row(V.BASE / "statistics" / V.LEVEL)
    model_old = {m: dict(value=float(ro[m]), se=float(ro[m + "_se"])) for m in V.METRICS}
    model_old["n"] = int(ro["n"])

    T = V.compare().set_index("subset")
    lab = {s: dict(n_cmp=int(T.at[s, "n_cmp"]),
                   n_only_new=int(T.at[s, "n_only_new"]),
                   n_only_old=int(T.at[s, "n_only_old"]),
                   mc_corr=float(T.at[s, "mc_corr"]),
                   mc_mad=float(T.at[s, "mc_mad"]),
                   mc_mad_se=float(T.at[s, "mc_mad_se"]),
                   mc_bias=float(T.at[s, "mc_bias"]))
           for s in ("TIGHT", "IDX", "OTHER")}

    return {"built_at": date.today().isoformat(), "px_effect": px,
            "model_old": model_old, "label_summary": lab, "notes": NOTES}


def main():
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장", OUT, "({:,} bytes)".format(OUT.stat().st_size))
    for s, g in d["px_effect"].items():
        print("  px {:6} n={:3} mad={:7.3f} se={} hi={}".format(
            s, g["n"], g["mad"],
            "   n/a" if g["se"] is None else "{:6.3f}".format(g["se"]),
            "   n/a" if g["hi"] is None else "{:6.3f}".format(g["hi"])))
    return d


if __name__ == "__main__":
    main()
