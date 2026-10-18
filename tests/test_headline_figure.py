# -*- coding: utf-8 -*-
"""헤드라인 그림 — 3스타에서 기준 모형(Direct+f+m)과 표형 벤치마크(XGBoost) 비교.

 README 가 내세우는 그림 한 장이다. 규약 세 가지를 지켜야 한다.
   1. 패널은 **R² 와 MAPE 두 개**만 (MAE·RMSE 등은 표로만 보고한다)
   2. 그림 안 글자는 **전부 영어**
   3. 막대는 **단일 모형 5시드 평균**, 오차막대는 같은 시드들의 SE
      (앙상블 예측으로 지표를 내면 막대와 오차막대가 서로 다른 대상이 된다)

 수치는 result/statistics/allstar_level.csv 의 group=3star 행에서 그대로 온다 —
 그림이 표와 다른 값을 말하면 안 된다.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import file_manager as fm      # noqa: E402

NAME = "headline_3star"
MODELS = ("Direct+f+m (1·2·3)", "XGBoost (1·2·3)")


def test_generator_declares_the_two_models_and_two_panels():
    import inspect
    import generate.eval_all_stars as EAS
    src = inspect.getsource(EAS)
    assert "headline_3star" in src, "헤드라인 그림 생성기가 없다"
    assert "HEADLINE" in src
    assert EAS.HEADLINE == MODELS
    assert EAS.HEADLINE_PANELS == ("R2", "MAPE"), "패널은 R²·MAPE 두 개로 고정한다"


@pytest.mark.skipif(not fm.stat("allstar_level").exists(), reason="성능 표 없음 (노트북 3 미실행)")
def test_source_rows_exist_for_the_figure():
    L = pd.read_csv(fm.stat("allstar_level"))
    g = L[(L.group == "3star") & (L.model.isin(MODELS))]
    assert len(g) == 2, "3스타 두 모형 행이 모두 있어야 한다"
    assert {"R2_se", "MAPE_se"} <= set(g.columns)
    d = g.set_index("model")
    assert d.at[MODELS[0], "R2"] > d.at[MODELS[1], "R2"], "기준 모형이 벤치마크보다 낮다"


@pytest.mark.skipif(not fm.image(NAME).exists(), reason="그림 없음 (노트북 3 미실행)")
def test_figure_is_produced():
    assert fm.image(NAME).stat().st_size > 20_000


def test_allstar_panels_use_r2_and_mape_not_mae():
    """7종 통합 그림의 성능 패널도 R²·MAPE 두 지표로 고정한다.

     종전에는 1·2·3스타 패널이 **MAE 막대**였다. 스타별로 가격대가 달라 원 단위 오차가
     직관적이긴 하지만, 성능 히스토그램은 R²·MAPE 로 통일하고 MAE·RMSE 등은 표로만 낸다
     (`allstar_level.csv` 에 여섯 지표가 모두 있다). 기울기 패널은 비율이라 그대로 둔다."""
    import inspect
    import generate.eval_all_stars as EAS
    src = inspect.getsource(EAS)
    assert EAS.STAR_PANELS == ("R2", "MAPE")
    assert "OOS MAE [KRW]" not in src, "스타별 패널에 MAE 막대가 남아 있다"
    assert "mean |1 - slope / MC slope|" in src, "기울기 패널은 유지한다"


def test_main_can_actually_call_headline_fig(tmp_path, monkeypatch):
    """**생성기를 실제로 호출한다.** 소스 문자열과 디스크 산출물만 보는 테스트는
     깨진 생성기가 남긴 그림을 보고 녹색을 내준다 — 실제로 그랬다.

     main() 이 `headline_fig(LV)` 를 부르는데 LV 가 어디에도 정의돼 있지 않아
     NameError 로 죽었고, 그래서 노트북 3 이 완주하지 못했다. 그림이 디스크에 있던 것은
     사람이 손으로 함수를 불렀기 때문이다."""
    import inspect
    import generate.eval_all_stars as EAS

    # (1) main() 안에서 headline_fig 에 넘기는 이름이 실제로 묶이는가
    src = inspect.getsource(EAS.main)
    assert "headline_fig(" in src, "main() 이 헤드라인 그림을 만들지 않는다"
    import ast
    tree = ast.parse(inspect.getsource(EAS))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    bound = {t.id for node in ast.walk(fn) for t in ast.walk(node)
             if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "headline_fig")
    for a in call.args:
        if isinstance(a, ast.Name):
            assert a.id in bound or hasattr(EAS, a.id),                 "main() 이 정의되지 않은 {} 를 headline_fig 에 넘긴다".format(a.id)

    # (2) 정의가 __main__ 가드보다 앞에 있는가 (스크립트 실행 시 미바인딩 방지)
    names = [(n.lineno, n.name) for n in tree.body if isinstance(n, ast.FunctionDef)]
    guard = [n.lineno for n in tree.body if isinstance(n, ast.If)
             and ast.unparse(n.test).replace(" ", "").startswith("__name__==")]
    if guard:
        hl = dict((nm, ln) for ln, nm in names)["headline_fig"]
        assert hl < guard[0], "headline_fig 가 __main__ 가드 뒤에 정의돼 있다"

    # (3) 실제로 그려지는가 — 가중치 없이 두 행짜리 합성 프레임으로
    monkeypatch.setattr(fm, "image", lambda n: tmp_path / (n + ".png"))
    LV = pd.DataFrame([
        {"group": "3star", "model": MODELS[0], "n": 100, "R2": .97, "R2_se": .001,
         "MAPE": .38, "MAPE_se": .01},
        {"group": "3star", "model": MODELS[1], "n": 100, "R2": .91, "R2_se": .001,
         "MAPE": .72, "MAPE_se": .01}])
    EAS.headline_fig(LV)
    assert (tmp_path / (NAME + ".png")).stat().st_size > 10_000


TERMS = "headline_3star_terms"


def test_raw_slope_csv_holds_numbers_not_lists():
    """시드별 가격을 리스트로 저장하면 CSV 소비자가 못 읽는다 — 평균을 적는다.

     per-seed 집계로 바꾸면서 `allstar_slope_raw.csv` 의 모형 열이 "[1.0, 2.0, ...]" 문자열이
     돼 버렸다. 기울기는 메모리에서 시드별로 내되, 파일에는 숫자를 적어야 한다."""
    import pandas as pd
    import pytest as _pt
    if not fm.stat("allstar_slope_raw").exists():
        _pt.skip("원자료 없음")
    C = pd.read_csv(fm.stat("allstar_slope_raw"))
    col = "Direct+f+m (1·2·3)"
    assert col in C.columns
    assert pd.api.types.is_numeric_dtype(C[col]), "모형 열이 숫자가 아니다 (리스트 문자열)"


def test_headline_terms_figure_uses_the_table_model():
    """구간평균 그림도 **표와 같은 모형**이어야 한다.

     종전에는 README 가 `Direct+f+m` 표 옆에 `Direct+m` 구간평균 그림을 실었다. 같은 문서에서
     두 모형을 한 대상인 양 보여주면 안 된다.

     참고: 구간평균은 **선형** 통계라 "시드 예측을 평균한 뒤 구간화" 와 "구간별로 평균한 뒤
     시드 평균" 이 같은 값이다. R²·MAPE 때와 달리 집계 규약은 문제가 아니고, 모형만 문제였다."""
    import inspect
    import generate.build_nb_viz as BNV
    src = inspect.getsource(BNV)
    assert "nb_pred_ball_dm.npy" in src, "노트북 3 이 기준 모형 예측을 읽지 않는다"
    assert "headline_3star_terms" in src, "기준 모형 구간평균 그림이 없다"


def test_notebook2_saves_the_headline_prediction():
    import inspect
    import generate.build_nb_training as BNT
    src = inspect.getsource(BNT)
    assert "nb_pred_ball_dm.npy" in src, "노트북 2 가 기준 모형 예측을 저장하지 않는다"


def test_only_one_generator_writes_each_figure():
    """같은 파일명을 두 생성기가 쓰면 **나중 것이 덮어쓴다.**

     실제로 그랬다: 노트북 3 의 cat1 절이 구간평균을 `headline_3star_terms.png` 로 저장한 뒤,
     나중에 도는 eval_all_stars 가 같은 이름으로 선 그림을 덮어썼다. 파일은 생겼고 로그도
     "saved" 라고 찍혀서 겉보기에는 성공이었다."""
    import pathlib as _p
    import re
    root = _p.Path(__file__).resolve().parent.parent
    # **실제 저장 호출만** 센다 — 이름이 문자열로 등장하는 것(경로 참조·주석)은 쓰기가 아니다.
    # 저장 호출만: savefig(fm.image("x")) 또는 save(fig, "x"). display(Image(fm.image("x")))
    # 같은 **읽기**는 제외한다.
    pat = re.compile(r'(?:savefig\(fm\.image\(|save\(fig,\s*)"([a-z0-9_]+)"')
    writers = {}
    for f in sorted(root.glob("generate/*.py")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for name in set(pat.findall(txt)):
            writers.setdefault(name, set()).add(f.name)
    dup = {n: sorted(w) for n, w in writers.items() if len(w) > 1}
    assert not dup, "한 그림을 여러 생성기가 쓴다: {}".format(dup)
