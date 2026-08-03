# -*- coding: utf-8 -*-
"""캐시 기초자산 종가(data/cache/px_*.parquet)가 yfinance 원본과 같은지 교차검증.

pytest 아님(네트워크 필요). 수동 실행:
    python tests/verify_px_yfinance.py                       # 기본 기간
    python tests/verify_px_yfinance.py 2015-01-05 2026-05-08

검증 내용
 ① 겹치는 날짜의 종가가 동일한가 (상대차)
 ② 5_segment 그림 11 과 같은 절차(180일 롤링 상관의 표본기간 평균)로 만든 상관행렬이 같은가
 ③ 180일 역사변동성 평균이 같은가
캐시가 yfinance 를 가공 없이 받은 것임을 확인하는 용도다(야후 원본 자체의 오류는 두 쪽 모두 공유).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from util import file_manager as fm   # noqa: E402

MAIN = ["^KS200", "^GSPC", "^STOXX50E", "^HSCE", "^N225"]
NAME = {"^KS200": "KOSPI 200", "^GSPC": "S&P 500", "^STOXX50E": "EURO STOXX 50",
        "^HSCE": "HSCEI", "^N225": "Nikkei 225"}
TOL_PX = 1e-6      # 종가 상대차 허용
TOL_CORR = 0.005   # 상관 절대차 허용
TOL_VOL = 0.05     # 변동성(%p) 허용


def logret(close):
    s = close.dropna().sort_index()
    return np.log(s[~s.index.duplicated()]).diff()


def corr_mat(X):
    """5_segment 그림 11 과 동일: 180일 롤링 상관 → 표본기간 평균."""
    C = X.rolling(180, min_periods=120).corr()
    return pd.DataFrame([[C.xs(a, level=1)[b].mean() for b in MAIN] for a in MAIN],
                        index=MAIN, columns=MAIN)


def vol_mean(X):
    return (X.rolling(180, min_periods=120).std() * np.sqrt(252)).mean()


def main(t0="2015-01-05", t1="2026-05-08"):
    import yfinance as yf

    cache = {t: pd.read_parquet(
        fm.CACHE / ("px_" + t.replace("^", "_").replace(".", "_") + ".parquet"))["close"]
        for t in MAIN}
    end = (pd.Timestamp(t1) + pd.Timedelta(days=1)).date().isoformat()
    raw = yf.download(MAIN, start=t0, end=end, auto_adjust=False, progress=False,
                      group_by="column", threads=True)
    assert raw is not None and not raw.empty, "yfinance 다운로드 실패 (네트워크/차단)"
    cl = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    miss = [t for t in MAIN if t not in cl.columns]
    assert not miss, f"받지 못한 티커: {miss}"
    fresh = {t: cl[t] for t in MAIN}

    print(f"기간 {t0} ~ {t1}")
    bad = []

    print("\n① 종가 일치도 (겹치는 날짜)")
    for t in MAIN:
        a, b = cache[t].dropna(), fresh[t].dropna()
        j = a.index.intersection(b.index)
        assert len(j) > 500, f"{NAME[t]}: 겹치는 날짜 {len(j)}일 (너무 적음)"
        rel = ((b.loc[j] - a.loc[j]).abs() / a.loc[j].abs()).max()
        ok = rel <= TOL_PX
        bad += [] if ok else [f"{NAME[t]} 종가 상대차 {rel:.2e}"]
        print(f"   {NAME[t]:15s} 겹침 {len(j):,}일 | 최대 상대차 {rel:.2e} {'OK' if ok else 'FAIL'}")

    XC = pd.DataFrame({t: logret(cache[t]) for t in MAIN}).sort_index().loc[t0:t1]
    XY = pd.DataFrame({t: logret(fresh[t]) for t in MAIN}).sort_index().loc[t0:t1]
    CC, CY = corr_mat(XC), corr_mat(XY)
    lab = [NAME[t] for t in MAIN]
    iu = np.triu_indices(len(MAIN), 1)
    dmax = float((CY - CC).abs().values[iu].max())
    print(f"\n② 180일 롤링 상관 평균 — 비대각 최대 차이 {dmax:.4f} "
          f"{'OK' if dmax <= TOL_CORR else 'FAIL'}")
    print(CY.set_axis(lab).set_axis(lab, axis=1).round(3).to_string())
    if dmax > TOL_CORR:
        bad.append(f"상관 최대 차이 {dmax:.4f}")

    oth = [t for t in MAIN if t != "^KS200"]
    for nm, Mx in (("캐시", CC), ("yfinance", CY)):
        ks = float(Mx.loc["^KS200", oth].mean())
        no = float(np.mean([Mx.loc[a, b] for i, a in enumerate(oth) for b in oth[i + 1:]]))
        print(f"   {nm:9s} KOSPI200 낀 쌍 {ks:.4f} | 안 낀 쌍 {no:.4f} | 차 {ks-no:+.4f}")

    V = pd.DataFrame({"cache": vol_mean(XC) * 100, "yfinance": vol_mean(XY) * 100})
    vmax = float((V.yfinance - V.cache).abs().max())
    print(f"\n③ 180일 역사변동성 평균(%) — 최대 차이 {vmax:.3f}%p "
          f"{'OK' if vmax <= TOL_VOL else 'FAIL'}")
    print(V.set_axis(lab).round(2).to_string())
    if vmax > TOL_VOL:
        bad.append(f"변동성 최대 차이 {vmax:.3f}%p")

    if bad:
        print("\nFAIL: " + " | ".join(bad))
        return 1
    print("\nPASS — 캐시는 yfinance 원본과 동일하고, 히트맵 절차도 재현된다.")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    raise SystemExit(main(*a) if a else main())
