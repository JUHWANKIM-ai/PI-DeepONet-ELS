# -*- coding: utf-8 -*-
"""파일 경로 중앙 관리. AbacusDirectoryPath 형식 차용 (클래스 기반 경로 + get_* 게터).

하위호환: 기존 코드가 쓰던 모듈 레벨 API(ROOT, dataset(), prediction() 등)는
싱글턴 PATH 를 감싸 그대로 제공하며, 전부 pathlib.Path 를 반환한다.
"""
import os
import glob
from pathlib import Path
from typing import Optional

ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ELSDirectoryPath:
    """저장소의 모든 경로를 한 객체에 모아 둔다 — 경로 문자열을 코드에 흩지 않기 위한 층.

    이 클래스의 게터는 **str** 을 돌려준다. 모듈 아래쪽의 하위호환 함수(`dataset()`,
    `image()` 등)가 그 결과를 `Path` 로 감싸므로, 호출부는 거의 항상 그쪽을 쓴다.
    `.exists()` 나 `/` 연산이 필요하면 모듈 함수를 써야 한다.
    """

    def __init__(self, base_directory: str) -> None:
        """경로 전부를 base_directory 기준으로 조립한다 (디렉터리는 만들지 않는다).

        입력
          base_directory : str — 저장소 루트. 보통 `ROOT_DIRECTORY`
        출력
          없음 (속성만 채운다). 실제 생성은 `ensure_dirs()` 가 한다
        """
        self.base_directory = base_directory

        ############################################################
        # Config
        ############################################################
        self.config_file_name = os.path.join(self.base_directory, "config.yaml")

        ############################################################
        # Data (원천 · 모델링 데이터셋 3종)
        ############################################################
        self.data_dir_name = os.path.join(self.base_directory, "data")
        self.raw_dir_name = os.path.join(self.data_dir_name, "raw")
        self.cache_dir_name = os.path.join(self.data_dir_name, "cache")
        self.source_file_name = os.path.join(self.data_dir_name, "els3_dataset.parquet")
        self.ml_dataset_file_name = os.path.join(self.data_dir_name, "ml.csv")
        self.deeponet_dataset_file_name = os.path.join(self.data_dir_name, "deeponet.csv")

        ############################################################
        # Result (예측 · 그림 · 통계)
        ############################################################
        self.result_dir_name = os.path.join(self.base_directory, "result")
        self.prediction_dir_name = os.path.join(self.result_dir_name, "predictions")
        self.image_dir_name = os.path.join(self.result_dir_name, "image")
        self.statistic_dir_name = os.path.join(self.result_dir_name, "statistics")

        ############################################################
        # Scratch
        ############################################################
        self.scratch_dir_name = os.path.join(self.base_directory, "scratch")

        # 데이터셋 이름 -> 경로
        self.dataset_files = {
            "ml": self.ml_dataset_file_name,
            "deeponet": self.deeponet_dataset_file_name,
        }

    ############################################################
    # Base / Config
    ############################################################
    def get_base_dir(self) -> str:
        """프로젝트 루트 경로 반환. 입력 없음 · 출력 str."""
        return self.base_directory

    def get_config_file(self) -> str:
        """config.yaml 경로 반환. 입력 없음 · 출력 str."""
        return self.config_file_name

    def ensure_dirs(self) -> None:
        """data·cache·result·scratch 하위 디렉토리를 모두 만든다 (있으면 넘어간다).

        입력 없음
        출력 없음 (파일시스템만 바꾼다)
        """
        for d in (
            self.data_dir_name,
            self.raw_dir_name,
            self.cache_dir_name,
            self.result_dir_name,
            self.prediction_dir_name,
            self.image_dir_name,
            self.statistic_dir_name,
            self.scratch_dir_name,
        ):
            os.makedirs(d, exist_ok=True)

    ############################################################
    # Data
    ############################################################
    def get_data_dir(self) -> str:
        """data 디렉토리 경로 반환. 입력 없음 · 출력 str."""
        return self.data_dir_name

    def get_raw_dir(self) -> str:
        """원천 CSV(raw) 디렉토리 경로 반환. 입력 없음 · 출력 str."""
        return self.raw_dir_name

    def get_cache_dir(self) -> str:
        """cache 디렉토리 경로 반환. 입력 없음 · 출력 str."""
        return self.cache_dir_name

    def get_source_file(self) -> str:
        """원천(피처 완성본) parquet 경로 반환. 입력 없음 · 출력 str."""
        return self.source_file_name

    def get_dataset_file(self, name: str) -> str:
        """모델링 데이터셋 경로 반환.

        입력
          name : "ml" | "deeponet"
        출력
          str — 해당 CSV 경로
        예외
          KeyError — 두 이름 외의 값
        """
        return self.dataset_files[name]

    ############################################################
    # Result
    ############################################################
    def get_result_dir(self) -> str:
        """result 디렉토리 경로 반환. 입력 없음 · 출력 str."""
        return self.result_dir_name

    def get_prediction_file(self, name: str) -> str:
        """모델별 예측 csv 경로 반환.

        입력
          name : str — 확장자 없는 이름. "<name>.csv" 로 붙는다
        출력
          str — result/predictions/<name>.csv. 디렉토리는 이 호출이 만든다
        """
        os.makedirs(self.prediction_dir_name, exist_ok=True)
        return os.path.join(self.prediction_dir_name, f"{name}.csv")

    def get_image_file(self, name: str) -> str:
        """그림(png) 경로 반환.

        입력
          name : str — 확장자 없는 이름. "<name>.png" 로 붙는다
        출력
          str — result/image/<name>.png. 디렉토리는 이 호출이 만든다
        """
        os.makedirs(self.image_dir_name, exist_ok=True)
        return os.path.join(self.image_dir_name, f"{name}.png")

    def get_statistic_file(self, name: str) -> str:
        """통계치(csv) 경로 반환.

        입력
          name : str — 확장자 없는 이름. "<name>.csv" 로 붙는다
        출력
          str — result/statistics/<name>.csv. 디렉토리는 이 호출이 만든다
        """
        os.makedirs(self.statistic_dir_name, exist_ok=True)
        return os.path.join(self.statistic_dir_name, f"{name}.csv")

    def list_prediction_files(self) -> list[str]:
        """저장된 예측 csv 경로 목록 반환. 입력 없음 · 출력 list[str] (이름 오름차순)."""
        return sorted(glob.glob(os.path.join(self.prediction_dir_name, "*.csv")))

    ############################################################
    # Scratch
    ############################################################
    def get_scratch_dir(self) -> str:
        """scratch 디렉토리 경로 반환. 입력 없음 · 출력 str."""
        return self.scratch_dir_name


# ============================================================================
# 모듈 싱글턴 + 하위호환 API (기존 코드는 fm.ROOT / fm.dataset() 등을 그대로 사용)
# 모든 반환값은 pathlib.Path 로 감싸 기존 .exists()/.glob()/relative_to()/"/" 를 지원.
# ============================================================================
PATH = ELSDirectoryPath(ROOT_DIRECTORY)

ROOT = Path(ROOT_DIRECTORY)
CONFIG = Path(PATH.config_file_name)
DATA = Path(PATH.data_dir_name)
RAW = Path(PATH.raw_dir_name)
CACHE = Path(PATH.cache_dir_name)
RESULT = Path(PATH.result_dir_name)
PRED = Path(PATH.prediction_dir_name)
IMAGE = Path(PATH.image_dir_name)
STAT = Path(PATH.statistic_dir_name)
SCRATCH = Path(PATH.scratch_dir_name)
DATASETS = {k: Path(v) for k, v in PATH.dataset_files.items()}


def ensure_dirs() -> None:
    """싱글턴 PATH 의 디렉토리 생성을 그대로 부른다. 입력 없음 · 출력 없음."""
    PATH.ensure_dirs()


def dataset(name: str) -> Path:
    """모델링 데이터셋 경로.

    입력
      name : "ml" | "deeponet"
    출력
      Path — 클래스 게터와 달리 Path 다. `.exists()` 를 바로 쓸 수 있다
    """
    return Path(PATH.get_dataset_file(name))


def rate(name: str) -> Path:
    """금리 커브 **원천** 경로 (data/raw/rates/).

    CD 91일(krw_curve.parquet 의 m3) · IRS par(irs_bondweb.parquet) ·
    NS 노드(krw_curve_fred.parquet)는 ECOS·본드웹·FRED 에서 받아오는 수집물이지
    우리 코드의 산출물이 아니다. Underlying 폴더에는 금리가 없어 원천에 따로 둔다.

     입력
       name : str — 파일명. 확장자까지 그대로 붙인다 (예: "krw_curve.parquet")
     출력
       Path — data/raw/rates/<name>. 존재 여부는 확인하지 않는다
    """
    return RAW / "rates" / name


def source() -> Path:
    """원천(피처 완성본) 경로. 입력 없음 · 출력 Path (data/els3_dataset.parquet)."""
    return Path(PATH.get_source_file())


def prediction(name: str) -> Path:
    """모델별 예측 csv 경로.

    입력
      name : str — 확장자 없는 이름 (예: "ball_dm_seed0")
    출력
      Path — result/predictions/<name>.csv. 디렉토리는 이 호출이 만든다
    """
    return Path(PATH.get_prediction_file(name))


def image(name: str) -> Path:
    """그림 경로.

    입력
      name : str — 확장자 없는 이름 (예: "headline_3star")
    출력
      Path — result/image/<name>.png. 디렉토리는 이 호출이 만든다
    """
    return Path(PATH.get_image_file(name))


def stat(name: str) -> Path:
    """통계표 경로.

    입력
      name : str — 확장자 없는 이름 (예: "allstar_level")
    출력
      Path — result/statistics/<name>.csv. 디렉토리는 이 호출이 만든다
    """
    return Path(PATH.get_statistic_file(name))


def list_predictions() -> list[Path]:
    """저장된 예측 csv 경로 목록. 입력 없음 · 출력 list[Path] (이름 오름차순)."""
    return [Path(p) for p in PATH.list_prediction_files()]
