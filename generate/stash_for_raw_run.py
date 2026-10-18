# -*- coding: utf-8 -*-
"""원천 전용 실행을 위한 산출물 격리 — 되돌림 수단을 남기는 이동.

`data/cache/` 전체 · `result/` 의 산출물 · `data/` 의 조립 데이터셋을 한 디렉터리로 옮겨,
**원천만 남은** 상태에서 노트북 4장이 전 과정을 다시 만드는지 본다.

남기는 것 (건드리지 않는다):
  data/raw/                     DART 공시 3종 + rates/ 금리 커브 3파일
  data/share_underlyings_v2/    기초자산 187 CSV (가격 + 배당수익률)

옮긴 디렉터리가 **유일한 되돌림 수단**이다. 검증이 끝나고 사람이 승인하기 전에는 지우지 않는다.
그래서 목적지가 이미 있으면 덮어쓰지 않고 거부한다.

실행: python generate/stash_for_raw_run.py scratch/raw_run_baseline
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from util import file_manager as fm  # noqa: E402

KINDS = ("cache", "models", "statistics", "predictions", "image", "datasets")
DATASET_FILES = ("els3_dataset.parquet", "els12_dataset.parquet", "ml.csv", "deeponet.csv")


def _move_dir_contents(src, dst):
    """src 안의 항목을 dst 로 옮기고 개수를 센다. **src 디렉터리 자체는 남긴다**.

    디렉터리를 지우면 파이프라인이 "경로가 없다" 로 죽어, 재현이 아니라 설정 오류가 된다.

    입력
      src : Path — 비울 디렉터리. 없으면 0 을 돌려주고 끝낸다
      dst : Path — 받을 디렉터리. 없으면 만든다
    출력
      int — 옮긴 항목 수 (파일과 디렉터리를 함께 센다)
    """
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        shutil.move(str(p), str(dst / p.name))
        n += 1
    return n


def stash(dst):
    """산출물을 dst 로 옮기고 범주별 개수를 돌려준다.

    입력
      dst : str | Path — 백업 디렉터리
    출력
      dict — KINDS 별 옮긴 개수 (cache · models · statistics · predictions · image · datasets)
    예외
      FileExistsError — dst 가 이미 있고 비어 있지 않을 때.
                        기존 백업을 덮어쓰면 되돌림 수단을 잃는다
    """
    dst = Path(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(
            "백업 디렉터리가 이미 있다: {} (되돌림 수단을 덮어쓸 수 없다)".format(dst)
        )
    dst.mkdir(parents=True, exist_ok=True)

    got = {k: 0 for k in KINDS}
    got["cache"] = _move_dir_contents(fm.CACHE, dst / "cache")
    for kind in ("models", "statistics", "predictions", "image"):
        got[kind] = _move_dir_contents(fm.RESULT / kind, dst / kind)
    (dst / "datasets").mkdir(parents=True, exist_ok=True)
    for name in DATASET_FILES:
        p = fm.DATA / name
        if p.exists():
            shutil.move(str(p), str(dst / "datasets" / name))
            got["datasets"] += 1
    return got


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/raw_run_baseline")
    n = stash(target)
    print("격리 -> {}".format(target))
    for k in KINDS:
        print("  {:12} {:4}개".format(k, n[k]))
    print()
    print("남은 원천:")
    for p in (fm.RAW, fm.DATA / "share_underlyings_v2"):
        cnt = sum(1 for x in p.rglob("*") if x.is_file()) if p.exists() else 0
        print("  {:34} {:4}개".format(str(p.relative_to(fm.ROOT)), cnt))
