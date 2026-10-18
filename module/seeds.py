# -*- coding: utf-8 -*-
"""공통난수 시드 — 상품 내용에서 결정적으로 유도한다.

**왜 행 번호가 아닌가.** 종전에는 `mc_seed = np.arange(len(df))`, 즉 발행일 정렬 후의 행 번호였다.
그러면 유니버스가 조금만 바뀌어도 그 뒤 상품의 시드가 전부 밀린다. 실제로 상품 123건이
새로 들어오자 공통 상품의 **98.2%(57,705/58,790)** 가 다른 난수를 받았고, 그 결과

  - 구본과 신본이 공통난수가 아니게 되어 **라벨 차이로는 재현을 판정할 수 없게** 됐다
    (입력이 완전히 같아도 독립 MC 추정 두 개의 차이만큼 평균 7.18원이 남는다)
  - 판정을 200건짜리 별도 측정(px_mad)으로 우회해야 했고, TIGHT 는 n=11 에 의존하게 됐다

`item` 의 안정적 해시를 쓰면 유니버스가 바뀌어도 기존 상품의 난수가 고정되므로, 앞으로는
전수 라벨을 비트 단위로 대조할 수 있다.

**왜 hashlib 인가.** 내장 `hash()` 는 문자열에 대해 프로세스마다 소금이 달라(PYTHONHASHSEED)
같은 상품이 실행마다 다른 시드를 받는다. 재현에 쓸 수 없다.

**충돌.** 58,913건을 2^31 공간에 넣으면 생일문제로 충돌이 더러 생긴다. 충돌은 두 상품이 같은
난수열을 쓴다는 뜻일 뿐, 서로 다른 계약·시장입력을 받으므로 편향을 만들지 않는다
(공통난수 자체가 원래 상품 간에 난수를 공유하는 기법이다).
"""
import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

SEED_MAX = 2**31 - 1  # mc_seed 는 int32 로 저장된다


def mc_seed_for(items: Iterable[object]) -> np.ndarray:
    """상품코드 목록 -> 공통난수 시드 배열.

    위치와 무관하고, 프로세스·플랫폼·파이썬 버전에 걸쳐 같은 값을 준다.

    입력
      items : 길이 n 의 반복자 — 상품 식별자(`item`). str() 로 바꿔 해시하므로
              숫자·문자가 섞여 있어도 된다
    출력
      ndarray(n,) int32 — sha1(item) 앞 8자리를 SEED_MAX 로 나눈 나머지.
                          0 이상 SEED_MAX 미만
    """
    return np.array(
        [int(hashlib.sha1(str(it).encode("utf-8")).hexdigest()[:8], 16) % SEED_MAX for it in items],
        dtype="int32",
    )


def seeds_stale(df: pd.DataFrame) -> bool:
    """저장된 데이터셋의 mc_seed 가 지금의 정의와 다른가 (열이 없으면 낡은 것으로 본다).

    "파일이 있는가" 가 아니라 "무엇으로 만든 파일인가" 를 묻기 위한 것이다. 시드 정의를
    바꿔도 저장본을 그대로 읽으면 라벨 재생성이 헛돈다 — 실제로 행 번호 시드가 남은 채
    노트북 0 이 재빌드를 건너뛰었다.

    유니버스가 늘어난 경우(새 상품)는 이것으로 잡히지 않는다. 그건 빌드해 봐야 알 수 있고,
    환경변수 REBUILD=1 로 강제한다.

    입력
      df : DataFrame — `item` 과 `mc_seed` 열을 본다
    출력
      bool — True 면 낡음(재생성 필요). 두 열 중 하나라도 없으면 **증명할 수 없으므로 True**
    """
    import numpy as _np

    if "mc_seed" not in df.columns or "item" not in df.columns:
        return True
    return not bool(
        _np.array_equal(
            _np.asarray(df["mc_seed"], dtype="int64"), mc_seed_for(df["item"]).astype("int64")
        )
    )
