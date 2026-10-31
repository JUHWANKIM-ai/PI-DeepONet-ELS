# 보관 KAN 코드 — Direct_KAN_20261031_v1

이 `KAN` 브랜치는 기존 ELS 프로젝트에 보관 KAN의 소스 코드를 추가한다.
기존 파일을 수정하지 않고 [KAN/](KAN/README.md)와 이 안내 문서만 추가했다.

- 금리 CNN, KAN branch/trunk 구조와 연결 기저함수
- NumPy 수식 평가기, PyTorch 체크포인트 로더, Vbase 계산과 추론 API
- 원래 모델·개선 구조 코드와 입력 순서·단위 규약
- 모델 구조 실행 예제와 원본 소스 SHA-256 대조표

**학습 가중치, 학습된 계수가 포함된 계산식, 데이터 및 학습·평가 기록은 포함하지 않는다.**
학습된 가격 추론에는 별도로 보관한 모델 파일이 필요하다.
후속 강건성 후보와 symbolic regression 결과로 보관 모델을 교체하지 않았다.

## 시작

```console
python -m pip install -r KAN/requirements-raw.txt
python KAN/architecture_example.py
```

위 예제는 구조와 입출력 크기를 확인하며 학습된 가격을 출력하지 않는다.
기존 전체 보관본을 별도로 가지고 있다면 다음 명령으로 가격 추론을 확인한다.

```console
python KAN/inference_example.py --model-dir PATH_TO_Direct_KAN_20261031_v1
```

## 보관 모델의 구조

- Branch: 금리10 → CNN32, VC10을 합쳐 `42 → 128 → 192 → 192`
- Trunk: 계약51과 Vbase1을 합쳐 `52 → 192 → 192 → 192`
- 두 출력의 내적으로 MC_B 가격을 직접 예측한다.
- Vbase는 입력으로만 쓰며 별도 KANO 보조망과 PDO는 없다.
- fold0/1: `RefineAdaptive`; fold2: `RefineTanh4`; fold3: `RefinePlain`

전체 설명과 파일 안내는 [KAN/README.md](KAN/README.md)에 있다.
