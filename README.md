# 오토콜 ELS 이론가 DeepONet 대리모델 (Surrogate Pricing)

- 한국형 **3-기초자산 worst-of 스텝다운 오토콜 ELS**의 **몬테카를로(MC) 이론가**를 재현하는 연산자학습(operator-learning) 대리모델
- 재시뮬레이션(10만 경로) 대비 가격 산출 속도
  - 200개 요청: **약 1,240배**
  - 북 전량 일괄: **약 36,500배**
  - 실측 기준이며, 배치 크기에 따라 배속 변동
- 금융감독 공시(DART) 및 시장데이터 기반 학습
- 정보누수를 방지한 **walk-forward** 프로토콜 적용
- 표형 머신러닝 벤치마크와 비교 평가

---

## 예측 대상

- 예측 타깃: **MC 이론가**
- 공시 공정가치는 학습 타깃에서 제외
- 정본 라벨: `mc_B`
- 라벨 생성: `generate/build_mc_variants.py`
- MC 엔진: `module/mc_engine.py`
- 가격 산출 방식: 일별 GBM + 촐레스키 기하 worst-of
- 변동성 보정: **무보정(k = 1)**
- 변동성 프리미엄 보정: 근거 및 통용성 부족으로 미채택

| 입력 | 현행 정의 | 구현 |
|---|---|---|
| 변동성 | 6개월(120 영업일) EWMA, λ = 0.99 | `module.features.vol_ewma_lam` |
| 상관 | KRX 영업일 120일 창, **쌍별** Pearson | `module.features.corr_bd` |
| 금리 | CD 91일 + IRS par 부트스트랩 제로커브, 드리프트·할인 양쪽 적용 | `module.curve` |
| 배당 | **연속 배당수익률 q**, LSEG 일별 지표 사용, 지수는 추종 ETF로 대체 | `module.dividend` |
| 경로 | **100,000**, 공통난수 `mc_seed` 적용 | `module.mc_engine.mc_daily_t` |

- 대리모델 입력: 바스켓 vol·corr + KRW 곡선 + 전체 계약
- 프라이서와 동일한 입력 사용
- forward 1회로 MC 이론가 재현
- µs 단위 추론

> **1단계 성능 평가 기준: MC 이론가 재현 정확도. 시장 공정가 지표는 별도 관리.**

---

## 결과

- 평가 대상: 3스타 ELS 시험구간 **24,350건**
- 평가 구성: 시드 5개 × 폴드 4개
- 막대: 단일 모형 5개의 평균
- 오차막대: 동일 시드들의 표준오차(SE)
- 앙상블 성능과 구분
- 그림: `result/image/headline_3star.png`

| 모형 | R² | MAPE | MAE | RMSE |
|---|---:|---:|---:|---:|
| **DeepONet (Direct + V_base + 단조)** | **0.9690 ± 0.0017** | **0.397%** | 38.4원 | 63.8원 |
| DeepONet (Direct + 단조), 기존 모형 | 0.9525 ± 0.0034 | 0.508% | 48.8원 | 78.9원 |
| XGBoost, 같은 입력 | 0.9098 ± 0.0013 | 0.724% | 69.5원 | 109.0원 |

### 비교 항목

- **V_base 투입 효과**
  - 비교 대상: 첫째 행과 둘째 행
  - 동일 아키텍처에서 프록시 입력 1열 추가
  - 상대오차: **0.508% → 0.397%**
- **아키텍처 효과**
  - 비교 대상: 첫째 행과 셋째 행
  - 공통 입력: 곡선 u0–9 + vol·corr + 전체 계약
  - R² 차이: 약 **5.9%p**
  - 상대오차 차이: 약 **1.8배**
- 헤드라인 그림: 기준 모형과 표형 벤치마크 2종으로 구성

### 계약조건별 가격 반응

- 평가 목적: 계약조건 변화에 대한 가격 반응이 MC 참값의 기울기와 일치하는지 확인
- 수준 지표와 기울기 지표 병행 평가
- 대상 모형: 위 표와 동일한 **Direct+f+m**
- 그림: `result/image/headline_3star_terms.png`
- 시각화 방식
  - 계약조건별 상품을 십분위(decile) 구간으로 분할
  - 구간평균 가격과 MC 참값 중첩
  - 비선형 반응 및 기울기 변화 확인
- 해석 기준: 절대 수준보다 **선의 모양과 MC 대비 차이**에 집중

| 요인 | 표본 수 | 관찰 결과 |
|---|---:|---|
| KI 배리어·쿠폰·첫 행사가 | 각 24,350건 | MC 참값과 거의 일치 |
| 스텝다운 폭 | OOS 전수 기준 | 0.15 부근에서 급락, 해당 구간의 상품구조 편중 반영 |
| 리자드 2개 요인 | 2,649건 | MC 대비 가격 수준 차이 발생 |

- 리자드 상품 비중: 3스타의 약 **11%**
- 리자드 관련 학습 표본 부족
- 단조손실 적용 범위: 리자드 지급률에 한정

### 기울기 평가 자료 구분

| 자료 | 구성 | 해석상 특징 |
|---|---|---|
| OOS 전수 분포 | 24,350건의 실제 상품 | 실제 상품 분포의 대표성 확보 |
| 섭동 격자 `allstar_slope.csv` | 스타당 기준상품 12개 | 계약조건을 직접 변경한 통제 실험 |

- 섭동 격자의 매끄러운 반응: 상품 간 이질성을 통제한 결과
- 인과적 반응 확인: 섭동 격자 활용
- 실제 상품 분포 확인: OOS 전수 자료 활용

### 전수 횡단면 OOS 잔차회귀

- 정본 통계: `result/statistics/xsec_slope.csv`
- 평가 대상: MC 참값 기울기가 유의한 16행
- 기준 모형 결과
  - 평균 `|배수 − 1|`: **0.114**
  - MC 참값과의 기울기 상관: **0.995**
- 참값 기울기가 0에 가까운 요인: 배수 대신 기울기 수준으로 해석
- 사유: 작은 분모로 인한 배수 불안정

### 섭동 격자 해석

- 자료: `allstar_slope.csv`
- 기준상품: 스타당 12개
- 모형 간 순위 판별에는 표본 부족
- 일부 요인에서 횡단면 통계와 부호 불일치
- 불일치 원인
  - 격자: 계약조건을 직접 변경한 반응 측정
  - 횡단면: 위험한 상품일수록 쿠폰이 높은 관계 등 교란 포함
- 측정 대상이 다르므로 두 통계의 혼합 해석 지양

### 전체 비교 자료

- 7종 모형 비교 및 1·2스타 확장 결과
  - `result/statistics/allstar_level.csv`
  - `result/image/allstar.png`

### 속도

- 측정 위치: `3_visualization` 3-5절
- 하드웨어: RTX 3060 Ti
- 비교 표본: 동일한 3스타 OOS 상품 200개
- 측정 범위: **가격 산출만**

| 방식 | 상품당 소요 시간 | MC 10만 경로 대비 |
|---|---:|---:|
| **MC 100,000 경로, 정본 라벨** | 66.1 ms | — |
| MC 40,000 경로 | 27.5 ms | — |
| 대리모형, 200개 요청 | 53.3 µs | **약 1,240배** |
| 대리모형, 전량 29,916개 일괄 | 1.8 µs | **약 36,500배** |

- 배치 크기에 따른 속도 차이
  - 200개 요청: 호출 오버헤드 비중 증가
  - 대규모 배치: 호출 오버헤드 분산
- 측정 제외 항목: EWMA·상관·커브 부트스트랩·배당 등 시장입력 조립
- 제외 사유: MC와 대리모형 양쪽에 공통으로 필요한 과정

### 속도 비교 기준

- 기준 경로 수: 정본 라벨과 동일한 **100,000경로**
- 적용 사유: 대리모형의 학습 대상과 비교 대상 일치
- 40,000경로 기준 적용 시
  - 배속 약 2.4배 과소평가
  - 학습 대상보다 잡음이 큰 가격과 비교

### 가격 일치도

- 상관계수: **0.981**
- MAE: **43.0원**
- RMSE: **74.5원**
- MAPE: **0.45%**

### 측정 프로토콜

- MC: 1회 측정
- 대리모형: 워밍업 3회 후 20회 측정의 중위값
- 측정 횟수의 비대칭 존재
- 배속 수치 해석 시 해당 조건 고려

### 산출물 위치

- 지표: `result/statistics/`
- 그림: `result/image/`

---

## 기준 모형의 입력 스키마 — `ball_dm` (Direct+f+m)

- 헤드라인 DeepONet에 해당하는 기준 모형
- Direct 타깃 계열 중 최고 성능

| 모집단 | n | R² | MAPE |
|---|---:|---:|---:|
| 3스타 | 24,350 | 0.9690 ± 0.0017 | 0.397% |
| 전체 1·2·3스타 | 29,916 | 0.9728 ± 0.0010 | 0.393% |

- `ball_d`와 동일한 타깃·손실 사용
- 차이점: 입력에 `V_base` 1열 추가
- 전체 R²: `ball_d` **0.9527** → `ball_dm` **0.9728**

```text
가중치  result/models/ball_dm_deeponet_anchor_seed{0..4}_fold{0..3}.pt
학습    module.train.train_curve_mono
신경망  module.networks.CurveOperatorV2
타깃    mc_B, 액면 1 기준 MC 이론가
방식    Direct 타깃, 잔차 차감 없음
```

### 입력 구성: 세 블록 + V_base

- 예측 1건당 입력: **실수 72개**

| 블록 | 차원 | 구성 |
|---|---:|---|
| `CURVE` | 10 | 제로금리 곡선 |
| `VC` | 10 | 변동성·상관·배당 |
| `CON` | 51 | 계약 조항 |
| `V_base` | 1 | 해석적 근사가 |

- `CON` 뒤에 `V_base` 결합
- `ball_dm` trunk 입력: **52차원**
- `ball_d` trunk 입력: **51차원**
- 세 입력 블록: 학습구간 통계 기반 z-정규화
- 구현: `zstats(..., tr)`
- 시험구간 통계 사용 제외

### ① `CURVE` (10) — 제로금리 곡선

- 발행일 KRW 제로금리를 10개 만기에서 추출
- 원천: CD 91일물 + IRS par 금리
- 커브 구성: 부트스트랩
- 구현: `module.curve`
- 적용 범위: 드리프트 및 할인

```text
만기(년) = 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10
```

- 처리 방식: **1D-CNN**
- 적용 사유
  - 만기 순서가 있는 신호
  - 인접 만기 간 높은 연관성

### ② `VC` (10) — 시장 상태

| 위치 | 내용 |
|---|---|
| 0–2 | 기초자산 변동성 3개, **오름차순 정렬** |
| 3–5 | 쌍별 상관 3개, **오름차순 정렬** |
| 6 | 바스켓 실효 변동성 `sig_eff` |
| 7–9 | 연속 배당수익률 q 3개, σ 정렬 순서 적용 |

- 정렬 목적: worst-of 페이오프의 자산 순서 대칭성 반영
- 자산 라벨 변경 시 가격 불변
- 정렬을 통해 신경망의 대칭성 학습 부담 축소

```text
sig_eff = mean(σ) × sqrt(1 + max(0, 1 − mean(ρ)))
```

- `sig_eff`: 바스켓을 단일 자산으로 근사한 실효 변동성
- 상관 하락 시 실효 변동성 증가
- worst-of에서 일부 자산의 하락 가능성이 커지는 효과 반영

### ③ `CON` (51) — 계약 조항

| 위치 | 이름 | 내용 |
|---|---|---|
| 0–11 | `strk_0..11` | 회차별 행사가, 조기상환 배리어 |
| 12–23 | `pmt_0..11` | 회차별 **누적** 지급률 |
| 24–35 | `lz_barr_0..11` | 리자드 배리어 |
| 36–47 | `lz_pmt_0..11` | 리자드 지급률 |
| 48 | `BARR_1/100` | KI 배리어 |
| 49 | `ANL_RTRN/100` | 연 쿠폰 |
| 50 | `TENOR` | 만기, 연 단위 |

- 12회차 미만 계약: 잔여 회차를 중립값으로 패딩
- 리자드 미포함 상품: `lz_barr = 1.2`
- `lz_barr = 1.2`: 중립값 및 리자드 미포함 표식
- `BARR_1/100 >= 1.0`: **항상 낙인** 의미
- 노낙인 상품과 구분 필요

### ④ `V_base` (1) — 해석적 근사가

- worst-of 바스켓을 단일 자산으로 근사한 간이 가격
- 닫힌식으로 계산
- 시뮬레이션 미사용

```text
V_base = Σ_j (C_{j-1} − C_j)(1 + pmt_j) · DF(θ_j)
       + C_K · DF(T) · [ p_KI · E[w_T | w_T < b_K] + (1 − p_KI) · 1.0 ]
```

- 구현: `module.vbase`
- 기반: 논문 식 19
- MC의 실제 ELS 페이오프에 맞춰 3개 항목 수정

| 항목 | 원문 | 적용 방식 |
|---|---|---|
| 만기 비상환·미낙인 | `1 + cT` | **원금 1.0** |
| 만기 손실 분기 | `w_0`로 대체 | **절단 로그정규 평균** `E[w_T │ w_T < b_K]` |
| `B >= 1.0` | — | `p_KI = 1`, 항상 낙인 |

### V_base의 역할

- 배당이 반영된 가격 기준선 제공
- 배당수익률 q의 기존 입력 위치: `VC`
- `CON`에는 배당수익률 q 미포함
- `V_base`를 trunk에 추가해 계약 측 입력에도 배당 반영
- 가격구조를 처음부터 학습하는 부담 축소
- 전체 R²: **0.9527 → 0.9728**

> `V_base`는 입력 피처로만 사용. 타깃에서 차감하지 않음.  
> 타깃에서 차감하는 `Resid+f+m`은 별도 계열이며, R²는 0.9796.

### 신경망 구조

```text
CURVE(10) ─ 1D-CNN ─┐
                     ├─ branch MLP ─→ (P=192)
VC(10) ─────────────┘                    │
                                         ⊙ 내적
CON(51) + V_base(1) ─ trunk MLP ─→ (P=192)
                                         │
                                      + b0 → 가격
```

- branch: 시장 상태 인코딩
- trunk: 계약조건 인코딩
- 출력: 두 잠재벡터의 내적 + 편향 `b0`
- 동일 시장에서 계약만 변경하는 경우 branch 재사용 가능

| 항목 | 값 |
|---|---:|
| 잠재차원 `P` | 192 |
| MLP 폭 | 128 |
| 반복 수 | 3,000 |
| 배치 크기 | 512 |
| 학습률 | 1e-3 |

---

## 단조손실 (`+m`) — 가격 반응 방향 제약

- 목적: 기대 방향이 명확한 계약조건 변화에 대한 부호 위반 억제
- 배치마다 계약조건 1종 섭동
- 원본 입력과 섭동 입력을 함께 forward
- 기대 방향을 위반한 만큼 벌점 부여

```python
viol = relu(margin − 기대부호 × (흔든가격 − 원가격)) × 적용마스크
mono_loss = viol.sum() / 적용마스크.sum()
```

- `relu` 기반 위반 교정
- margin 조건 충족 시 벌점 0
- 기준 설정인 margin 0에서는 방향 위반만 교정

### 섭동 방향 5종

| 항목 | 섭동 폭 | 기대 부호 | 적용 대상 |
|---|---|---|---|
| `coupon` | +0.005, 쿠폰·누적지급률을 같은 비율로 변경 | **+** | 전체 |
| `strike` | +0.02, 12회차 전체 | **−** | 전체 |
| `barrier` | +0.02, KI 배리어 | **−** | `B < 0.99`인 낙인형 |
| `lz_pmt` | ×1.25, 리자드 지급률 | **+** | 리자드 포함 회차 |
| `lz_barr` | +0.02, 리자드 배리어 | **−** | 리자드 포함 회차 |

### 적용마스크

- 리자드 미포함 상품: 리자드 관련 손실 계산에서 제외
- 포함 조건: `lz_barr < 1.2`인 회차가 하나 이상 존재
- 제외 상품: 손실 분자·분모 모두 미포함
- 목적: 가격 반응이 발생하지 않는 상품의 불필요한 벌점 방지

### 기준 모형 설정

```python
MONO = dict(fixed_w=1.0, kinds=["lz_pmt"], margin_krw=0.0)
```

| 항목 | 설정 | 적용 사유 |
|---|---|---|
| `kinds` | **`["lz_pmt"]`만 적용** | 기준 모형에서 부호 위반이 발생한 항목 |
| `margin_krw` | **0.0원** | 최소 반응 크기 없이 부호만 제약 |
| `fixed_w` | **1.0 고정** | 자동 산정 시 `ml → 0` 부근에서 가중치 발산 |

- 나머지 4개 항목: 단조손실 없이도 기대 부호 충족
- 불필요한 손실 추가 시 데이터손실과의 경쟁 발생
- 실제 위반 항목에만 제약 적용

### margin 0원 설정 근거

- `margin > 0`: 부호와 최소 반응 크기를 동시에 요구
- 요구 반응이 MC 참값보다 클 경우 구조적 과보정 발생
- 학습 섭동: 리자드 지급률 ×1.25
- 섭동 격자에서 측정한 상품별 MC 반응

| 스타 | 상품별 반응 [원] | 8원 미만 |
|---|---|---:|
| 1★ | 2.3 · 3.3 · 6.4 · 6.6 · 8.0 · 10.8 | **5/6** |
| 2★ | 3.0 · 5.2 · 5.2 · 5.4 · 12.5 · 28.2 | **4/6** |
| 3★ | 1.3 · 21.0 · 22.3 · 26.1 · 29.3 · 46.9 | 1/6 |

- 1·2스타: 실제 반응이 대부분 8원 미만
- 종전 `margin_krw = 8.0` 설정
  - 실제 반응이 6원인 상품에도 최소 8원 요구
  - 리자드 지급률 기울기 2배 이상 과대 추정
  - 1스타 기울기 배수: **2.13**
- 3스타: 대부분 21~47원 반응
- 3스타에서는 hinge 활성화가 드물어 margin 8원과 0원의 차이 제한적
- 종전 8원 기준: 3스타 결과 중심으로 선정

### margin 변경 결과

- 비교 조건: 5시드, 동일 폴드, margin만 변경

| 지표 | margin 8원 | **margin 0원** |
|---|---:|---:|
| 리자드 지급률 기울기 배수, 1★ | 2.13 | **0.93** |
| 리자드 지급률 기울기 배수, 2★ | 2.16 | **1.13** |
| 평균 \|1−배수\|, 1·2·3★ 평균 | 0.246 | **0.125** |
| R², ALL | 0.9739 ± 0.0006 | 0.9728 ± 0.0010 |
| MAPE, ALL | 0.384% | 0.393% |

- 수준 성능 변화는 제한적
- 리자드 지급률 기울기 개선
- 나머지 5개 요인 변화는 소수점 둘째 자리까지 거의 없음
- 해당 요인: KI 배리어·쿠폰·행사가·스텝다운·리자드 배리어
- `lz_pmt`에만 손실을 적용한 실험 구성과 일치

> 리자드 지급률 섭동 비교 표본은 스타당 6개. 개선 방향은 확인되나, 0.93과 1.00의 유의한 차이 여부는 판별 곤란. 확정 판단에는 OOS 전수 횡단면 재평가 필요.

### 추가 MC 계산

- 섭동 입력에 대한 MC 라벨 추가 생성 없음
- 섭동 입력의 MC 참값 미사용
- 가격 반응 부호에 대한 사전 지식만 활용
- 추가 MC 시뮬레이션 비용 없음

---

## 예측 1건 생성

```python
from module import star as ST, train as TR

D  = ST.make_all()                 # CURVE · VC · CON · MC
vb = ST.vbase_all(D)               # V_base, 1열
Dx = copy.copy(D)
Dx.CON = np.concatenate([D.CON, vb[:, None]], 1)   # 51 → 52

f = TR.load_curve_forward("result/models/ball_dm_deeponet_anchor_seed0_fold0.pt")
price = f(Dx.CURVE[[i]], Dx.VC[[i]], Dx.CON[[i]])  # 액면 1 기준
```

- 예측 활용 시 시드 5개의 평균 사용 권장
- 단일 시드 결과에 따른 모형 순위 판정 지양
- 신경망 시드 간 R² 변동: 약 0.0006
- 성능 보고 기준: **5시드 평균 ± SE**

---

## 레포 구조

```text
config.yaml              # 전체 하이퍼파라미터의 단일 출처
pyproject.toml           # black, 100컬럼 포매터 설정 / 패키지 설치용 구성 제외
requirements.txt         # 의존성 버전 고정 / Python 3.11, conda env `abacus`

0_data_preprocessing.ipynb  # raw → data/els3_dataset.parquet / 탈락 퍼널·분포·시계열
1_MC_pricer.ipynb           # 현행 프라이서 / 시장입력·비트단위 재현검증·경로잡음
2_model_training.ipynb      # 통합·3스타 전용 대리모형 학습 / OOS 수준 지표
3_visualization.ipynb       # 스타별 성능·계약조건 반응·프라이서·상품 시각화

module/     27파일        # 데이터·특성·MC 엔진·네트워크·학습·추론
model/       9파일        # 모델 플러그인 / 레지스트리 자동발견
util/        7파일        # 경로·설정·시드·지표·그림·속도
generate/   15파일        # 노트북·라벨·표·그림 생성 및 재현 판정
tests/      50파일        # pytest / 262 passed · 29 skipped

data/                    # 원천 입력 및 파생 데이터 / git-ignore
result/                  # predictions/·image/·statistics/·models/ / git-ignore
                         # README 수록 그림 2장만 추적

docs/                    # 설계 노트·계획서 / git-ignore
paper/                   # 참고 논문 / git-ignore
scratch/                 # 임시 작업물 / git-ignore
```

- 데이터·가중치·중간결과: Git 추적 제외
- 산출물: 노트북에서 재생성
- 코드 외 추적 대상
  - README 수록 그림 2장
  - 재현 판정 근거 `tests/fixtures/raw_run_verdict.json`

### 데이터 — 원천 입력과 재생성 대상

- 원천 입력 위치: **2곳**
- `data/cache/`: 입력이 아닌 산출물
- 그 외 `data/` 파생물: Git 추적 제외
- 검증 방식: 캐시·가중치·결과표 제거 후 원천만으로 노트북 4장 재실행
- 검증 결과: 전체 파이프라인 재생성 확인

| 경로 | 내용 | 관리 방식 |
|---|---|---|
| `data/raw/` | DART 공시 CSV, 금리커브 3파일, `udly_ticker_map.json` | 100MB 초과, Git 추적 제외, 로컬 보관 |
| `data/share_underlyings_v2/` | 기초자산 187개 CSV, 일별 종가·배당수익률, `master.csv` | 원천 입력 |
| `data/cache/` | MC 라벨, `V_base`, 중간 매핑 | 재생성 산출물 |
| `data/els3_dataset.parquet` | 3스타 데이터셋 | 노트북 0에서 재생성 |
| `data/els12_dataset.parquet` | 1·2스타 데이터셋 | 노트북 0에서 재생성 |
| `data/ml.csv` | 모델링 데이터셋 | 노트북 0에서 재생성 |
| `data/deeponet.csv` | DeepONet 데이터셋 | 노트북 0에서 재생성 |

- DART 공시 CSV: `AUTO_CALL`, `SCHD_INFO`, `UDLY_INFO`
- 금리커브 파일: CD 91일·IRS par·NS 노드
- 기초자산 가격: 액면분할만 조정한 `Close_Adj` 사용
- 배당: 엔진 드리프트에서 별도 차감
- 배당 역조정 가격 사용 시 배당 이중계산 발생

---

## 재현 절차

- 필요 자료: 원천 데이터
- 실행 순서: 노트북 **0 → 1 → 2 → 3**
- 환경: Python 3.11, conda env `abacus`

```bash
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute --inplace 0_data_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace 1_MC_pricer.ipynb
jupyter nbconvert --to notebook --execute --inplace 2_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace 3_visualization.ipynb
```

- 별도 CLI 단계 없음
- 각 노트북에서 필요한 산출물 직접 생성
- 기존 산출물 존재 시 생성 단계 생략

| 노트북 | 산출물 | 실측 비용 |
|---|---|---|
| **0** | `els3_dataset.parquet` 59,037건 · `els12_dataset.parquet` 15,752건 · `ml.csv` · `deeponet.csv` · 그림 3장 | 약 3분 |
| **1** | `mc_variants12` 15,752건 · `mc_variants` 59,037건 · `mc_check_allstar` · `mc_boot_curve_200` | **약 160분** |
| **2** | 가중치 140개, 7종 × 5시드 × 4폴드 · `nb_training_level.csv` | 약 25분 |
| **3** | 그림 24장 · 표 13개 | 약 10분 |

### 재생성 설정

- 노트북 0·1: `REBUILD = True`
- 노트북 2: 환경변수 `RETRAIN=1`
- 라벨 변경 시 저장 가중치 재학습 필요

### 샤드 캐시 의존성 해소

- 종전 구조
  - 노트북 0의 `mc` 열이 `data/cache/mccal_shard_*.json` 28개에 의존
  - 캐시 부재 시 단일코어 기준 수 시간 재시뮬레이션 필요
- 현행 구조
  - `module.data.build_datasets`: `mc`·`recent_margin` 열 없이 데이터셋 생성 가능
  - `module.mcvar`: 학습 라벨 `mc_B` 직접 공급
  - 샤드 캐시 없이 원천 자료만으로 0→3 실행 가능

### 공통 관리

- 하이퍼파라미터: `config.yaml`
- 경로: `util/file_manager.py`
- `combine`: 고비용 단계로 별도 분리
- 나머지 과정: 노트북 0→3 순서로 상단부터 실행

---

## 코드 맵

- 노트북 생성기: `generate/build_nb_*.py`
- 노트북 직접 수정 시 다음 생성 과정에서 덮어쓰기 발생
- 지속적으로 반영할 변경: 생성기 수정 필요

### `module/` — 파이프라인 본체

| 파일 | 역할 |
|---|---|
| `build_source.py` | DART 공시 → `els3_dataset.parquet`, 적격 판정·탈락 사유 집계 |
| `underlying.py` | `share_underlyings_v2` 가격 어댑터, `Close_Adj` 및 UDLY_ID 대체 해석 |
| `mc.py` | 가격 적재의 단일 출처, `returns_by_ticker`·`PX_SOURCE` 전환 |
| `dividend.py` | 배당 소스 분기 통합, `div_kwargs`·지수→ETF 프록시 |
| `curve.py` | CD + IRS par 부트스트랩 제로커브 |
| `features.py` | EWMA 변동성·영업일 쌍별 상관·곡선 노드 |
| `market.py` | 상품별 시장입력 조립, `state_for` → 엔진 kwargs `dkw` |
| `mc_engine.py` | 일별 GBM 기하 worst-of, torch GPU 벡터화 |
| `seeds.py` | `item` 해시 기반 공통난수 시드, 행 번호 미사용 |
| `mcvar.py` | 라벨 적재, `make_D`에서 `D.MC`에 `mc_B` 공급 |
| `star.py` | 1·2·3스타 통합 `make_all`, `vbase_all` |
| `data.py` | `ml.csv`·`deeponet.csv` 생성 및 라벨 열 구성 |
| `train.py` | `train_curve_mono`, MSE + 단조 hinge |
| `networks.py` | `CurveOperatorV2`, branch=곡선 CNN+vol·corr, trunk=계약 |
| `vbase.py` | 닫힌식 기반 준해석적 프록시 `V_base` |
| `contract.py` | 계약 섭동 격자, `GRID`·`apply_factor` |

### `generate/` — 생성기 및 검증 도구

| 파일 | 역할 |
|---|---|
| `build_nb_{data,pricer,training,viz}.py` | 노트북 0·1·2·3 생성 |
| `build_mc_variants.py` | MC 라벨 산출 및 출처 스탬프, `stamp`·`step_stale` |
| `eval_all_stars.py` | 7종 × 1·2·3스타 성능·기울기 평가 및 헤드라인 그림 |
| `eval_cross_section.py` | 전수 횡단면 잔차회귀 기울기 |
| `verify_raw_run.py` | 원천 전용 재현 판정, 부분집합별 비교 및 2표본 검정 |
| `freeze_verdict.py` | 재현 판정 근거의 저장소 픽스처 고정 |

### `util/` — 공통 유틸리티

- `file_manager.py`: 전체 경로 관리
- `plot.py`: 그림 규약 관리
  - dpi 400
  - 영문 표기
  - PPT 밴드
- `metric.py`: 평가 지표 통합 계산
  - R²·MAE·RMSE·MAPE·Bias·Spearman

---

## 코드 규약

- 적용 범위: `module`·`model`·`util`·`generate`
- 대상 함수: 356개

| 항목 | 규약 | 확인 방법 |
|---|---|---|
| 포매터 | **black, 100컬럼** | `black --check module model util generate tests` |
| 독스트링 | 함수 **356/356**, 클래스 **8/8**, 한국어 작성 | — |
| 타입 힌트 | 파라미터 **849/849**, 반환 **356/356** | `typing.get_type_hints()` 전수 통과 |
| 테스트 | 50파일, **262 passed · 29 skipped**, 약 42초 | `pytest` |
| 그림 표기 | 전체 영문 | `util/plot.py` |

### 독스트링

- API 설명과 설계 메모 병행
- 필수 기재 항목
  - 기능 요약 1줄
  - 설계 이유 및 기각 대안
  - `입력`·`출력` 블록
  - `None` 반환 가능 조건
- 입력 부족 시 처리
  - 예외 대신 `None` 반환
  - 호출부에서 해당 상품 탈락 처리

```python
def div_kwargs(
    tks: Sequence[str], t0: pd.Timestamp, ten: float, src: str = DEFAULT_SRC
) -> tuple[dict[str, Any], np.ndarray, int] | None:
    """(엔진 kwargs, qcont(3), 이산 이벤트수). LSEG 수익률 부재 시 None 반환.

    종전 generate/build_mc_variants.py에 중복된 배당 분기 통합.

    입력
      tks : list[str] 길이 3 — 기초자산 티커
      src : "lseg" | "yf" — 배당 소스
    출력
      (dkw, qcont, nev) | None
        dkw : dict — "lseg"이면 {"qcont": q}, "yf"이면 {"divlog": D}
        LSEG 수익률 확보 불가 시 None 반환
    예외
      ValueError — src가 SRCS 범위 밖인 경우
    """
```

### 타입 힌트

- `from __future__ import annotations` 미사용
- 타입 주석 즉시 평가 방식 적용
- import 시 타입 표현식 평가를 통해 오류 조기 확인
- 예측기 호출 규약을 타입으로 구분

| 예측기 | 타입 |
|---|---|
| 행 인덱스 기반 | `Callable[[npt.ArrayLike], np.ndarray]` |
| 임의 입력 forward | `Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]` |

- 두 호출 규약의 혼용 금지

### 테스트

- 과거 실제 결함 1건당 테스트 파일 1개 구성
- 예외 없이 수치만 잘못 산출되는 결함 중심
- 주요 사례
  - 배당이 0으로 입력된 가격
  - 부분 기록된 라벨 파일을 완료 파일로 인식
  - 동일 파일명으로 그림 중복 저장
- 목적: 육안으로 발견하기 어려운 수치 오류의 재발 방지

---

## 설정 (`config.yaml`)

| 항목 | 설정 내용 |
|---|---|
| `seed` | 난수 시드 |
| `device` | 연산 장치 |
| `data.walk_forward` | 폴드 경계 `[0.6, 0.7, 0.8, 0.9, 1.0]` |
| 시험구간 | 미래 40%, 확장 윈도우 4폴드 |
| `val_frac` | 검증구간 비율 |
| `time_decay` | 시간 감쇠 |
| `train` | 연산자망 반복 수·학습률·early stopping |
| `networks.P` | 잠재차원 |
| `tabular` | 표형 벤치마크 하이퍼파라미터, `xgb`·`lgbm`·`gbm`·`ridge` |

---

## 참고문헌

- 출판사 저작물인 논문 PDF: 저장소 업로드 제외
- `paper/`: Git 추적 제외
- 인용 정보: 아래 목록으로 관리

1. Lu et al. “Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.” *Nature Machine Intelligence* 3.3 (2021): 218–229.
2. Wang, Wang, Perdikaris. “Learning the solution operator of parametric PDEs with physics-informed DeepONets.” *Science Advances* 7.40 (2021): eabi8605.
3. Lee, Huh, Jeong. “DeepONet-based surrogate modeling for bond option pricing.” *AIMS Mathematics* 11.3 (2026): 5853.
4. Andreou, Han, Li. “Stock options pricing via machine learning methods combined with firm characteristics.” (2023).
5. Jiao et al. “Solving forward and inverse PDE problems on unknown manifolds via physics-informed neural operators.” *arXiv:2407.05477* (2024).