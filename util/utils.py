# -*- coding: utf-8 -*-
"""범용 헬퍼: config 로딩, 시드 고정, 디바이스 선택."""
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from . import file_manager as fm

_CONFIG = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """config.yaml 로딩 (1회 캐시).

    입력
      path : str | Path | None — 설정 파일 경로. None 이면 `fm.CONFIG`(저장소 루트 config.yaml)
    출력
      dict — 파싱된 설정 전체. 모듈 전역 `_CONFIG` 에도 보관한다
    """
    global _CONFIG
    if path is None:
        path = fm.CONFIG
    with open(path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config() -> dict[str, Any]:
    """캐시된 설정을 돌려주고, 아직 없으면 그때 한 번 읽는다.

    입력
      없음
    출력
      dict — `load_config` 와 같은 구조
    """
    return _CONFIG if _CONFIG is not None else load_config()


def set_seed(seed: int = 0) -> None:
    """random·numpy·torch 의 전역 시드를 한꺼번에 고정한다.

    세 난수원을 호출부마다 따로 고정하면 한 곳을 빠뜨린다 — 학습 쪽은 이 함수만 쓴다.
    MC 공통난수(`module.seeds.mc_seed_for`)와는 별개다: 그쪽은 상품별 경로 재현,
    이쪽은 가중치 초기화·배치 순서 재현이다.

    입력
      seed : int — 세 라이브러리에 같은 값으로 넣는다 (학습 시드는 0~4)
    출력
      없음 (전역 난수 상태만 바꾼다)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(cfg: dict[str, Any] | None = None) -> str:
    """학습·추론에 쓸 torch 디바이스 문자열을 고른다.

    입력
      cfg : dict | None — 설정. None 이면 `get_config()`.
            `cfg["device"]` 가 "cpu"/"cuda" 면 그대로 따르고, "auto"(기본)면 가용성으로 정한다
    출력
      str — "cpu" 또는 "cuda"
    """
    cfg = cfg or get_config()
    d = cfg.get("device", "auto")
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"
