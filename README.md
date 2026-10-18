# 오토콜 ELS 이론가 DeepONet 대리모델 (Surrogate Pricing)

한국형 **3-기초자산 worst-of 스텝다운 오토콜 ELS(주가연계증권)**의 **몬테카를로 이론가(MC 이론가)**를
재현하는 연산자학습(operator-learning) 대리모델 **DeepONet**. 재시뮬레이션(10만 경로) 대비
**약 1,240배**(200개 요청) ~ **36,500배**(북 전량 일괄) 빠르게 가격을 산출한다 — 실측값이며
배치 크기에 좌우된다. 금융감독 공시(DART)와 시장데이터를 결합해 학습하고, 정보누수 없는 **walk-forward**
프로토콜로 강력한 테이블형 머신러닝 벤치마크와 비교 평가한다.

---

## 무엇을 예측하는가

타깃은 공시 공정가치가 아니라 **MC 이론가**다. 정본 라벨은 `mc_B`(`tools/build_mc_variants.py`)이고,
엔진은 `module/mc_engine.py` 의 일별 GBM + 촐레스키 기하 worst-of 이며 **무보정(k = 1)** 이다
— 변동성 프리미엄 보정을 탐색했으나 근거·통용성이 약해 기각했다. 시장입력 정의는 아래와 같다.

| 입력 | 현행 정의 | 구현 |
|---|---|---|
| 변동성 | 6개월(120 영업일) EWMA, λ = 0.99 | `module.features.vol_ewma_lam` |
| 상관 | KRX 영업일 120일 창, **쌍별** Pearson | `module.features.corr_bd` |
| 금리 | CD 91일 + IRS par 부트스트랩 제로커브 (드리프트·할인 양쪽) | `module.curve` |
| 배당 | **연속 배당수익률 q** (LSEG 일별 지표, 지수는 추종 ETF) | `module.dividend` |
| 경로 | **100,000** (공통난수 `mc_seed`) | `module.mc_engine.mc_daily_t` |
학습된 DeepONet은 이 가격을 forward 한 번으로 재현하므로 모델은 **MC 대리모델**이다: 프라이서와 동일한
입력(바스켓 vol·corr + KRW 곡선 + 전체 계약)을 받아 µs 단위로 추론한다.

두 모델 계열을 비교한다.

- **직접(단일 단계)** — 테이블형 회귀기가 `ml.csv` 전체 특성으로 `mc` 를 한 번에 예측.
- **하이브리드(2단계)** — `y = MC_hat(stage-1 앵커) + residual_hat(stage-2)`. stage-2 타깃은
  `mc − MC_hat`(앵커 자신의 근사오차)이며 시장 마진 항은 쓰지 않는다.

> `model/` 디렉토리는 **플러그인 레지스트리**(`model/registry.py`)다. 물리기반 연산자
> (PI-DeepONet / BS-PINN)를 나중에 `run(D, cfg)` 함수를 가진 `model/*.py` 파일 하나로 끼워 넣으면
> 파이프라인 수정 없이 자동 편입된다. `model/_template.py` 참고.

---

## 결과 — walk-forward OOS (시험 n = 29,916; 시드 5개 × 폴드 4개 평균 ± SE)

**1단계 성능은 MC 이론가 재현만으로 재며**, 시장 공정가 지표는 섞지 않는다. 전량 재실행으로
`result/statistics/allstar_level.csv` 에서 그대로 옮긴 값이다.

| 모델 | 바꾼 것 | R² | MAPE % | MAE 원 | RMSE 원 | bias 원 |
|---|---|---:|---:|---:|---:|---:|
| **Direct+m (1·2·3)** | — (**기준 모형**) | **0.9554** ± 0.0043 | 0.520 | 49.8 | 79.4 | −14.3 |
| Direct+m (3스타만) | 학습 풀 | 0.8985 ± 0.0062 | 0.716 | 68.4 | 120.2 | −35.1 |
| XGBoost (1·2·3) | 아키텍처 | 0.8998 ± 0.0011 | 0.776 | 73.9 | 119.6 | −19.1 |
| XGBoost (3스타만) | 아키텍처 + 학습 풀 | 0.8402 ± 0.0013 | 0.944 | 90.2 | 151.1 | −43.1 |
| Resid+f+m (1·2·3) | 잔차 타깃 + `V_base` 피처 | **0.9788** ± 0.0008 | 0.348 | 33.4 | 55.0 | −2.4 |
| Resid+f+m (3스타만) | 위 + 학습 풀 | 0.9648 ± 0.0012 | 0.433 | 41.4 | 70.8 | −14.2 |
| Direct+f+m (1·2·3) | `V_base` 피처 | 0.9739 ± 0.0006 | 0.384 | 36.9 | 61.1 | −0.6 |

**핵심 결과 세 가지.**

1. **통합 학습이 사는 것은 커버리지다.** 전체(ALL)로는 R² 0.9554 대 0.8985 로 통합이 앞서지만,
   그 차이는 3스타를 더 잘 맞혀서가 아니다. 스타별로 쪼개면 분명해진다.

   | | n | 통합 학습 | 3스타 전용 |
   |---|---:|---:|---:|
   | 1스타 | 1,977 | **0.9487** ± 0.0062 | 0.2833 ± 0.0475 |
   | 2스타 | 3,589 | **0.9573** ± 0.0062 | 0.8523 ± 0.0185 |
   | 3스타 | 24,350 | 0.9540 ± 0.0044 | 0.9534 ± 0.0054 |

   **3스타에서는 둘이 구별되지 않는다**(0.09 결합 SE). 통합 모형이 ALL 에서 이기는 이유는
   3스타 전용 모형이 1스타에서 R² 0.28 로 **붕괴**하기 때문이다 — 본 적 없는 구조이기 때문이다.
   즉 한 모형으로 1·2·3스타를 다 커버하는 것이 이득이지, 3스타를 더 잘 맞히는 것이 아니다.

   3스타에서 **어느 쪽이 앞서는지는 보고하지 않는다.** 라벨 잡음에 따라 뒤집히기 때문이다 —
   유니버스 58,913건일 때는 3스타 전용이 0.9557 대 0.9460 으로 앞섰고, 59,037건으로 늘린 뒤에는
   구별되지 않는다. 재현되는 것은 "3스타에서 비슷하고 1·2스타에서 통합이 압도한다" 뿐이다.
   (`result/image/cat1_{1,2,3}star_level.png`)

2. **DeepONet 이 XGBoost 를 앞선다.** 같은 입력(곡선 u0-9 + vol·corr + 전체 계약)에서 0.9554 대 0.8998 이다.
   다만 기준 모형의 SE(0.0043)가 XGBoost(0.0011)의 4배다 — 신경망은 시드 간 흔들림이 크므로
   0.9554 는 한 점이 아니라 구간으로 읽어야 한다.
3. **프록시를 피처로 주면 더 오른다.** `V_base`(해석적 근사)를 넣으면 0.9739, 잔차 타깃까지 쓰면 0.9788 이다.
   단일 시드로 순위를 말하지 않는다 — 위 값은 모두 시드 5개 평균이고 R²·MAPE 에 SE 를 함께 적었다.

**기울기.** 수준 지표만으로는 부족하다. 전수 횡단면 OOS 잔차회귀에서 요인별 기울기를 MC 참값과
나란히 재며(`result/statistics/xsec_slope.csv`), 참값 기울기가 유의한 구간에서 기준 모형의
기울기 배수는 평균 |배수 − 1| = 0.065 다. 참값 기울기가 0 근처인 요인은 배수의 분모가 폭발하므로
기울기 수준으로 본다.

**속도** (`3_visualization` 3-5절 실측, RTX 3060 Ti · 같은 3스타 OOS 200개 · **가격 산출만**):

| | 상품당 | MC 10만 대비 |
|---|---|---|
| **MC 100,000 경로** (정본 라벨) | 66.2 ms | — |
| MC 40,000 경로 | 27.5 ms | — |
| 대리모형 (Direct+단조, 5시드, 200개 요청) | 53.3 µs | **약 1,240배** |
| 대리모형 (같은 구성, 전량 29,916개 일괄) | 1.8 µs | **약 36,500배** |

배속은 배치 크기에 크게 좌우된다 — 200개 요청은 호출 오버헤드가 지배하고, 큰 배치에서는 분산된다.
시장입력 조립(EWMA·상관·커브 부트스트랩·배당)은 두 경로 모두 똑같이 필요하므로 측정에서 제외했다.
**비교 기준은 정본과 같은 10만 경로다** — 대리모형이 재현하려고 배운 가격이 그것이기 때문이다.
4만 경로를 기준으로 삼으면 배속이 2.4배 과소평가되고, 대리모형을 배운 적 없는 더 시끄러운 가격과
비교하게 된다. 가격 일치는 corr 0.981 · MAE 43.0원 · RMSE 74.5원 · MAPE 0.45%.
**측정 프로토콜은 비대칭이다** — MC 는 1회 측정, 대리모형은 워밍업 3회 뒤 20회 중위값이다.
MC 쪽을 반복 측정해도 µs 단위 결론은 바뀌지 않지만, 배속 수치는 그 비대칭 위에 있다.

지표: `result/statistics/`  ·  그림: `result/image/`.

---

## 레포 구조

```
config.yaml              # 모든 하이퍼파라미터의 단일 출처
0_data_preprocessing.ipynb  # raw + cache -> data/els3_dataset.parquet + 탈락 퍼널 · 분포 · 시계열
1_MC_pricer.ipynb           # 현행 방법론 프라이서: 시장입력 · 재현검증 · 경로잡음 · 공정가 격차
2_model_training.ipynb      # 대리모형 학습(1·2·3스타 통합 / 3스타 전용) + OOS 수준 지표
3_visualization.ipynb       # 그림 일체 4개 범주: 스타별 성능 · 계약조건 반응 · 프라이서 · 상품

model/                   # 모델 플러그인(레지스트리 자동발견): benchmark, deeponet, stage2, _template
module/                  # 데이터 + 특성/MC 엔진 + 네트워크 + 학습 + 추론
util/                    # file_manager(경로), utils(설정/시드), metric, plot, speed
data/                    # cache/ 와 raw/ 만 추적 대상 입력(아래 참고)
result/                  # predictions/, image/, statistics/, models/(폴드별 저장 가중치)
paper/                   # 참고 논문 [1]-[5]
```

### 데이터 — 추적 대상 vs 재생성 대상

입력은 **원천 두 곳뿐**이다. `data/cache/` 는 입력이 아니라 **산출물**이며, `data/` 의 나머지도 파생물로
git-ignore 된다. 이 구분은 실제로 검증했다 — 캐시·가중치·결과표를 전부 치우고 원천만 남긴 상태에서
노트북 4장이 전 과정을 다시 만든다.

- `data/raw/` — DART 공시 CSV(`AUTO_CALL`, `SCHD_INFO`, `UDLY_INFO`) · `rates/` 금리커브 3파일
  (CD 91일 · IRS par · NS 노드) · `udly_ticker_map.json`. **git-ignore**(>100 MB); 로컬 보관.
- `data/share_underlyings_v2/` — 기초자산 187 CSV(일별 종가 + 배당수익률) + `master.csv`.
  가격은 `Close_Adj`(액면분할만 조정)를 쓴다. 엔진이 배당을 드리프트에서 따로 빼므로
  배당 역조정 계열을 쓰면 이중계산이 된다.
- `data/cache/` — **산출물.** MC 라벨(`mcvar/mc_variants*.parquet`), 프록시 `V_base`, 중간 매핑.
- `data/els3_dataset.parquet`, `els12_dataset.parquet`, `ml.csv`, `deeponet.csv` — 노트북 0 이 **재생성**.

---

## 재현 — 원천 자료만 있으면 노트북 4장으로 끝난다

```bash
pip install -r requirements.txt          # Python 3.11 (conda env `abacus`)

jupyter nbconvert --to notebook --execute --inplace 0_data_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace 1_MC_pricer.ipynb
jupyter nbconvert --to notebook --execute --inplace 2_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace 3_visualization.ipynb
```

별도 CLI 단계는 없다. 각 노트북이 자기 산출물을 만들고, **이미 있으면 건너뛴다**.

| 노트북 | 만드는 것 | 실측 비용 |
|---|---|---|
| **0** | `els3_dataset.parquet`(59,037) · `els12_dataset.parquet`(15,752) · `ml.csv` · `deeponet.csv` · 그림 3 | 약 3분 |
| **1** | `mc_variants12`(1·2스타 15,752) · `mc_variants`(3스타 59,037) · `mc_check_allstar` · `mc_boot_curve_200` | **약 160분** |
| **2** | `models/` 가중치 140개(7종 × 5시드 × 4폴드) · `nb_training_level.csv` | 약 25분 |
| **3** | 그림 23장 · 표 12개 (7종 비교 · 전수 횡단면 기울기 · 속도) | 수 분 |

다시 만들려면 노트북 0·1 은 `REBUILD = True`, 노트북 2 는 환경변수 `RETRAIN=1`
(라벨이 바뀌면 저장된 가중치는 낡는다).

**샤드 캐시는 더 필요하지 않다.** 종전에는 노트북 0 의 `mc` 열이 `data/cache/mccal_shard_*.json`
28개에 묶여 있었고 그게 없으면 단일코어 수 시간을 재시뮬해야 했다. 지금은 `module.data.build_datasets`
가 `mc`·`recent_margin` 열이 없어도 데이터셋을 만들고, 학습 라벨은 `module.mcvar` 가 `mc_B` 로
직접 공급한다. 그래서 원천만으로 0→3 이 끝까지 돈다.

모든 하이퍼파라미터는 **`config.yaml`**, 모든 경로는 **`util/file_manager.py`** 로 관리한다.
`combine` 단계만 분리돼 있는 것은 그것이 유일한 고비용 단계이기 때문이며, 그 외 노트북 0→3 은 위에서
아래로 순차 실행된다.

---

## 패키지 레퍼런스 — `model/`

`run(D, cfg) -> {name: DataFrame}` 를 노출하는 모든 `model/*.py` 는 `2_run` 이 자동발견·실행한다.
모델 추가 = 파일 하나 넣기(`_` 로 시작하는 파일은 발견에서 제외).

### `model/registry.py`
플러그인 발견·실행.
- `discover()` → `run(D, cfg)` 를 노출하는 `model` 하위모듈들의 `(name, run)` 목록(이름 정렬).
  `registry` 자기 자신, `_` 접두 파일, `run` 없는 모듈은 제외.
- `run_all(D, cfg, verbose=True)` → 각 모델을 실행해 예측 dict 를 병합. 모델명 중복 시 `ValueError`.

### `model/benchmark.py`
테이블형 ML 모델 전체 — 전부 공정가가 아니라 `mc` 를 예측.
- **직접 벤치마크**(단일 단계): `bench_ridge`, `bench_gbm`, `bench_lgbm`, `bench_xgboost`,
  `bench_catboost`. `predict_direct_tab` 로 `ml.csv` 전체 regime 특성 위에서 학습.
- `_xgb_anchor(D, cfg, tr, va, te, save_path=None)` — stage-1 XGB 앵커. **DeepONet 앵커와 동일한 블록**
  (`[곡선 u0-9 | vol·corr | 계약]`)으로 `mc` 를 예측(순수 아키텍처 비교). `predict(idx)` 콜백 반환·`.pkl` 저장.
- `run(D, cfg)` → 직접 5종 + `xgb_hybrid`(2단계 XGB 앵커 + XGB 잔차).

### `model/deeponet.py`
DeepONet-Curve 하이브리드(2단계, MC 타깃, 마진 없음).
- `_anchor` / `_anchor_l1` / `_anchor_mape` — `train_curve` 로 `D.MC` 를 MSE / L1 / MAPE 손실로
  학습하는 stage-1 `CurveOperatorV2` 앵커. 각각 `predict(idx)` 콜백 반환.
- `run(D, cfg)` → `deeponet_hybrid`(MSE 앵커 + XGB 잔차), `deeponet_hybrid_l1`, `deeponet_hybrid_mape`,
  `deeponet_hybrid_s2don`(MSE 앵커 + **DeepONet** 잔차 `don_resid`).

### `model/stage2.py`
공용 stage-2 잔차모델(하이브리드끼리 stage-1 앵커만 다름). 입력은 deeponet 특성 블록
`D.DON = [곡선 u0-9 | vol·corr·sig_eff | 계약]`.
- `xgb_resid(D, cfg, tr, va, te, target, ...)` — 기본 stage-2: `D.DON` 위 XGBoost
  (`deeponet_hybrid`/`_l1`/`_mape`·`xgb_hybrid` 사용). `load_xgb_resid(D, path)` 로 재로드.
- `don_resid(D, cfg, tr, va, te, target, ...)` — DeepONet stage-2: `MarginOperator`
  (branch = vol·corr·sig_eff, trunk = `[곡선 | 계약]`), train 폴드 표준화 + early stopping
  (`deeponet_hybrid_s2don` 사용). `load_don_resid(D, path)` 로 재로드.

### `model/_template.py`
새 플러그인 템플릿. 계약 규약(필수 예측 컬럼 `ITEM_CD, isu_ord, y_true, y_pred`; 하이브리드는
`mc_true, mc_pred, resid_true, resid_pred` 추가), `D` 의 필드, 현행 MC-타깃 관례
(`predict_hybrid(..., target=D.MC, use_margin=False)`)를 문서화.

---

## 패키지 레퍼런스 — `module/`

### `module/data.py`
데이터 스키마·생성·로딩 — 특성 그룹 정의의 단일 출처.
- 상수: `VOLCORR`(sig1-3, rho12/13/23, sig_eff), `UC`(곡선 u0-9), `CONTRACT`(strk0-11 + 배리어 +
  쿠폰 + 만기), 그리고 `BASE`/`REG`/`CAT`(테이블형)·타깃/앵커 이름.
- `build_datasets()` → **`data/ml.csv`**(테이블형 `BASE+REG+CAT`)와 **`data/deeponet.csv`**
  (연산자 입력: 곡선 + vol·corr·sig_eff + 계약 + r) 생성. 둘 다 `mc` / `fair` / `recent_margin` 포함.
- `load(cfg)` → `SimpleNamespace` `D`: `MC`, `FAIR`, `rm`, `ITEM`, `ORD`, branch/trunk 배열
  `VC`(n,7) / `CURVE`(n,10) / `CON`(n,15), stage-2 블록 `DON`(n,32), walk-forward 폴드.
- `walk_forward(n, bounds, val_frac, val_seed)` → 확장윈도우 `[(tr, va, te), …]` 폴드. 각 train
  윈도우에서 랜덤 validation 부분집합을 떼어냄(early stopping 용).
- 헬퍼: `time_weights`(반감기 365.25일 시간감쇠 가중), `to_tensor`, `zstats`(train 폴드 표준화), `featnum`.

### `module/build_source.py`
raw + cache 로 base 데이터셋 재유도(MC 는 여기서 계산하지 않음).
- `build_source(save=True, verbose=True)` → raw DART 테이블 + 캐시 가격/곡선 로드, 후보
  3-기초자산 **STEP / KRW / fair∈[0.70,1.05] / tenor∈[0.5,5]** 오토콜 필터, 180일 vol·corr,
  KRW NS 곡선 노드, 계약 특성·모멘텀 유도 후 `data/els3_dataset.parquet` 에 `fair` +
  `recent_mktvol`(인과적 90일 이동평균 변동성)와 함께 저장.

### `module/features.py`
공용 특성 계산(`0_data` 와 MC 엔진 간 DRY).
- `nsb`, `zero_curve`, `krw_beta`, `krw_curve_nodes` — KRW Nelson-Siegel 곡선 적합·노드 추출.
- `vol180(ret, dt)`, `corr180(rets, dt)` — 기준일 이전 180거래일 역사 변동성 / 3자산 상관.
  `chol_psd(corr)` — 고윳값 클리핑 PSD 보정 폴백이 있는 촐레스키.

### `module/mc.py`
MC 이론가 프라이싱 엔진(재산출 노트북·샤드 스크립트 공용).
- `load_market()` → 로그수익률 시계열, KRW 곡선, 상품→기초자산 / 행사가 매핑(`u3map`/`strk_by` 캐시).
  `load_kmap()` → 보정맵 또는 `None`(**정본 = 무보정**).
- `mc_daily(...)` — 촐레스키 상관 충격 일별 GBM, 일별 KI/오토콜 관측, KRW NS 할인(청크 단위).
  `price_one(mk, kmap, item, iord, B, c, ten, sig_eff, …)` → 상품 1건 프라이싱,
  `(mc, vol1-3, rho12/13/23, r_krw, k)` 반환. 상수: `NPATH`(40,000), `MC_COLS`.

### `module/mc_shards.py`
전량 MC 재산출을 독립 샤드로 병렬화(Windows OOM 방지 위해 `ProcessPool` 회피).
- CLI `main()`: `shard <k> <N>` 은 `tasks[k::N]` 프라이싱 → `data/cache/mccal_shard_<k>.json`
  (200개마다 `scratch/mc_progress/shard_<k>.json` 에 진행 heartbeat 기록); `monitor <N> [초]` 는 전
  샤드 heartbeat 를 합산해 누적 진행률·처리율·ETA·완료샤드 수를 한 줄로 갱신 표시; `combine <N>` 은
  전 샤드를 병합해 `els3_dataset.parquet` 에 `mc` + MC 입력 컬럼 + `recent_margin` 기록(+ heartbeat
  정리); `test` 는 앞 30행을 단일프로세스로 검증.

### `module/networks.py`
DeepONet 아키텍처.
- `CurveOperatorV2(nvc, ncon, P)` — stage-1 앵커: branch = 곡선 1D-CNN + vol·corr 융합,
  trunk = 계약 MLP; 내적 헤드.(스팟 입력 없음 — 스칼라 가격이 타깃.)
- `MarginOperator(nb, nt, P)` — stage-2 잔차 DeepONet: branch(시장상태)·trunk(계약) 내적 + 학습형 bias.
  `mlp(d, p)` — 공용 3층 Tanh MLP 빌더.

### `module/train.py`
stage-1 앵커 학습.
- `train_curve(D, cfg, tr, te, target, va=None, loss="mse", return_predict=False, save_path=None)` —
  표준화된 곡선/vol·corr/계약 특성 위에서 `CurveOperatorV2` 를 `target`·`loss`
  (`mse`/`l1`/`huber`/`mape`)로 학습. validation early stopping + `.pt` 저장. train/test 예측(및 선택적
  `predict(idx)` 콜백) 반환.
- `load_curve_predictor(D, path)` — 가중치 로드 → forward 전용 `predict(idx)`.
- `_opt`, `_EarlyStop`(best 가중치 복원, validation 기반), `_loss_fn` — `model/stage2.py` 도 재사용.

### `module/pipeline.py`
walk-forward OOS 예측 조립 + 폴드별 가중치를 `result/models/<name>_fold{k}.{pt|pkl}` 로 저장.
- `predict_direct_tab(D, cfg, model_key, name=None, target=None)` — 폴드별 단일단계 벤치마크
  (target 기본값 `D.MC`).
- `predict_hybrid(D, cfg, anchor_fn, resid_fn=None, name=None, target=None, use_margin=True)` — 2단계
  조립: 앵커 학습 → 잔차 타깃 `hybrid_residual_target(target, mc_hat, rm)`(stage-1 **예측값** 기준이라
  학습·추론 일치) → 잔차모델 학습/로드(기본 `xgb_resid`) → `y_pred = mc_hat + margin + resid`.
  MC 파이프라인은 `target=D.MC, use_margin=False` 로 호출.

### `module/tabular.py`
직접 벤치마크와 XGB stage-2 가 공용하는 테이블형 회귀기.
- `fit_tab(D, cfg, model, tr, te, feat, target, tw=True, va=None, save_path=None, return_predictor=False)`
  — `ridge`/`gbm`/`lgbm`/`cat`/`xgb` 를 시간감쇠 가중·early stopping(xgb/lgbm/cat)으로 적합, `.pkl` 저장.
  `load_tab_predictor(D, path)` 로 재로드 → `predict(idx)`.

### `module/infer.py`
저장 가중치로 **공정가(FAIR) 2단계 하이브리드** 예측 재현. 최종목표가 MC 이론가로 바뀐 뒤
현행 노트북 4장은 이 경로를 쓰지 않는다 — 가중치·예측은 `scratch/result_unused/` 에 있다.
- `_DIRECT`(직접 5종)·`_HYBRID`(2단계 5종) 레지스트리.
- `predict_all_from_weights(D, cfg)` → 각 모델의 폴드 가중치를 로드해 OOS test 인덱스를 forward,
  파이프라인과 동일 스키마의 DataFrame 반환(`y_true = mc`; 하이브리드는 단계별 R² 용 컬럼도 포함).

---

## `util/`

- `file_manager.py` — 중앙 경로 관리(`fm.dataset("ml"|"deeponet")`, `fm.image`, `fm.stat`,
  `fm.prediction`, `fm.RESULT/"models"`).
- `utils.py` — `load_config()`, `set_seed()`, 디바이스 선택.
- `metric.py` — `metrics()`(R²/MAE/RMSE/MAPE/Bias/Spearman)·`stage_r2()`(하이브리드 stage-1/2/final).
- `plot.py` — 학술/PPT 그림 기본값(import 시 rcParams 를 **dpi 400 · 투명배경 · PPT 콘텐츠밴드
  13.33″** 로 설정): `metric_bars`, `scatter_grid`, `stage_r2_bar`, `speed_bar`.
- `speed.py` — 전 10개 모델(직접 + 하이브리드)의 추론 지연 측정.

---

## 설정 (`config.yaml`)

`seed`, `device`; `data.walk_forward` 폴드 경계 `[0.6,0.7,0.8,0.9,1.0]`(test = 미래 40%, 확장 4폴드),
`val_frac`, `time_decay`; `train`(연산자망 반복수 / lr / early stopping); `networks.P`(잠재차원);
벤치마크·stage-2 공용 `tabular` 트리 하이퍼파라미터(`xgb`/`lgbm`/`cat`/`gbm`/`ridge`).

---

## 참고문헌 (PDF 는 `paper/`)

1. Lu et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." *Nature Machine Intelligence* 3.3 (2021): 218–229.
2. Wang, Wang, Perdikaris. "Learning the solution operator of parametric PDEs with physics-informed DeepONets." *Science Advances* 7.40 (2021): eabi8605.
3. Lee, Huh, Jeong. "DeepONet-based surrogate modeling for bond option pricing." *AIMS Mathematics* 11.3 (2026): 5853.
4. Andreou, Han, Li. "Stock options pricing via machine learning methods combined with firm characteristics." (2023).
5. Jiao et al. "Solving forward and inverse PDE problems on unknown manifolds via physics-informed neural operators." *arXiv:2407.05477* (2024).
