# 보관 KAN 소스 코드

`Direct_KAN_20261031_v1`의 모델 구조와 독립 추론 소스를 보관한 폴더다.
가중치·학습된 수식 계수·시장 데이터·학습 기록은 포함하지 않는다.

## 구조 확인

```console
python -m pip install -r KAN/requirements-raw.txt
python KAN/architecture_example.py
```

구조 예제는 학습되지 않은 모델로 크기와 수치 계산의 정상 여부만 확인한다.
그 출력을 학습된 가격으로 해석하지 않는다.

## 기존 학습 모델로 추론

기존 전체 보관본을 별도로 가지고 있다면 다음과 같이 실행한다.

```console
python KAN/inference_example.py --model-dir PATH_TO_Direct_KAN_20261031_v1
```

이 명령은 지정한 보관본의 예제 입력과 수식 파일을 읽는다. 해당 파일이 없으면 실행할 수 없다.
GitHub 코드만으로 학습된 가격을 재현할 수 있는 배포 묶음은 아니다.

## 소스 구성

| 파일 | 역할 |
|---|---|
| `runtime/els_kan_frozen/kan_models.py` | KAN 구조와 기저함수 |
| `runtime/els_kan_frozen/predictor.py` | 독립 추론 API |
| `runtime/els_kan_frozen/formula_numpy.py` | NumPy 계산식·미분 평가 |
| `runtime/els_kan_frozen/checkpoints.py` | 체크포인트 로딩 |
| `runtime/els_kan_frozen/_frozen/` | 금리 CNN, 입력 처리, Vbase 계산 |
| `model_sources/` | 보관 당시 모델·개선 구조의 원래 구현 |
| `input_schema.json` | 입력 순서와 단위 |
| `source_integrity.json` | 복사한 코드와 원본의 SHA-256 대조표 |

금리 CNN의 32개 특징과 VC10을 branch KAN에, 계약51과 Vbase를 trunk KAN에 넣는다.
Branch는 `42→128→192→192`, trunk는 `52→192→192→192`다.
두 출력의 내적으로 MC_B 가격을 직접 학습하며 Vbase는 입력에만 사용한다.
KANO 보조망과 PDO는 없다.

fold0/1은 adaptive3, fold2는 tanh4, fold3은 tanh3 기저를 선택했다.
원본 구조 후보에는 Gaussian RBF 기반 원 논문과 달리 tanh 계열 등의 별도 구현이 포함된다.
후속 symbolic regression 후보로 이 소스를 교체하지 않았다.

이 폴더는 구조·추론 코드의 공개 범위다. 원래 학습을 재현하려면 ELS 프로젝트의
학습 코드와 별도로 보관한 데이터·환경을 준비해야 한다.
