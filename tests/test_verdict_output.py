# -*- coding: utf-8 -*-
"""판정 출력은 기본 콘솔에서 죽지 않아야 하고, 세 상태를 세 단어로 구분해야 한다.

실측: cp949 콘솔(윈도우 기본)에서 `UnicodeEncodeError: '—'` 로 죽었다. CSV 를 먼저 쓰고
그 다음에 죽으므로, 이걸 게이트로 걸면 **인코딩 때문에 '실패' 로 끝난다.**

그리고 verdict 는 통과/미결론/기준없음 세 상태를 돌려주는데 main 은 두 단어만 찍었다 —
게이트가 걸렸으나 **결론을 못 낸** IDX 가, 애초에 게이트가 없는 OTHER 와 같은 말로 보였다.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not fm.stat("allstar_level").exists(), reason="성능 표 없음")
def test_main_survives_a_cp949_console():
    """윈도우 기본 콘솔에서도 돌아야 한다 — 출력이 게이트의 종료코드를 좌우한다."""
    env = dict(os.environ, PYTHONIOENCODING="cp949")
    env.pop("PYTHONUTF8", None)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "generate", "verify_raw_run.py")],
        capture_output=True,
        cwd=ROOT,
        env=env,
    )
    assert r.returncode == 0, r.stderr.decode("cp949", "replace")[-400:]


def test_three_states_get_three_distinct_marks():
    """통과 · 미결론 · 기준없음 이 서로 다른 말로 찍혀야 한다."""
    import inspect
    import generate.verify_raw_run as V

    src = inspect.getsource(V.main)
    assert "미결론" in src, "미결론을 기준없음과 같은 말로 찍고 있다"
    assert src.count('"기준없음"') <= 1


def test_frozen_rows_do_not_print_a_sentinel_n():
    """고정 근거 행의 n = -1 은 사람이 읽는 표에 들어가면 안 된다."""
    import inspect
    import generate.verify_raw_run as V

    assert "n=-1" not in inspect.getsource(V._frozen_table).replace(" ", "")
