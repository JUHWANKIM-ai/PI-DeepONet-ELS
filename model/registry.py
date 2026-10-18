# -*- coding: utf-8 -*-
"""model/ 플러그인 자동발견. model/*.py 가 `run(D, cfg) -> {name: DataFrame}` 규약만 지키면
여기서 자동 발견·실행된다. 새 모델은 파일 하나만 model/ 에 넣으면 2_run 에 자동 편입.
규약·입력(D) 설명은 model/_template.py 참고."""
import importlib
import pkgutil
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pandas as pd

import model as _pkg

_SKIP = {"registry"}  # 언더스코어(_template 등)로 시작하는 모듈은 자동 제외


def discover() -> list[tuple[str, Callable[..., dict[str, pd.DataFrame]]]]:
    """run(D, cfg) 를 노출하는 model 하위 모듈을 모아 돌려준다.

    `_` 로 시작하는 모듈(`_template` 등)과 `_SKIP` 의 이름은 건너뛴다. 모듈을 실제로
    import 하므로, 그 모듈의 최상위 코드가 무거우면 이 호출이 무거워진다.

    입력
      없음 (model 패키지 전체를 훑는다)
    출력
      list[(str, callable)] — (모듈 이름, run 함수), 모듈 이름 오름차순.
                              run 이 없거나 호출 가능하지 않은 모듈은 빠진다
    """
    found = []
    for mi in pkgutil.iter_modules(_pkg.__path__):
        if mi.name.startswith("_") or mi.name in _SKIP:
            continue
        mod = importlib.import_module(f"model.{mi.name}")
        fn = getattr(mod, "run", None)
        if callable(fn):
            found.append((mi.name, fn))
    return sorted(found, key=lambda x: x[0])


def run_all(
    D: SimpleNamespace, cfg: dict[str, Any], verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """발견된 모든 모델의 run(D, cfg) 를 실행해 예측을 하나의 사전으로 병합한다.

    모델 이름이 겹치면 **즉시 멈춘다** — 조용히 덮어쓰면 뒤에 실행된 모듈의 예측이
    앞선 모듈의 이름으로 저장된다.

    입력
      D       : Dataset — `module.data.load` 의 반환 (규약은 model/_template.py)
      cfg     : dict — `util.utils.load_config()`
      verbose : bool — True 면 모듈별로 만든 예측 이름을 찍는다
    출력
      dict[str -> DataFrame] — 예측 이름에서 walk-forward OOS 예측 표로
    예외
      ValueError — 두 모듈이 같은 예측 이름을 낼 때
    """
    preds = {}
    for name, fn in discover():
        out = fn(D, cfg)
        dup = preds.keys() & out.keys()
        if dup:
            raise ValueError(
                f"중복 모델 이름 {sorted(dup)} (module 'model.{name}'). 예측 이름은 유일해야 합니다."
            )
        preds.update(out)
        if verbose:
            print(f"[model.{name}] -> {list(out.keys())}")
    return preds
