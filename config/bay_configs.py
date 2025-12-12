# -*- coding: utf-8 -*-
"""
Bay별 통합 설정

명명 규칙:
- DAG: {task_type}_{bay_id}_dag (예: observation_12bay_dag)
- Queue:
  - Observer: obs_{bay_id} (예: obs_12bay)
  - Processor: proc_{bay_id} (예: proc_12bay)
- Worker:
  - Observer: observer-{bay_id} (예: observer-12bay)
  - Processor: processor-{bay_id} (예: processor-12bay)

새로운 Bay 추가 시 이 파일만 수정하면 됩니다.
"""

from typing import Dict, Any, List
import os
import numpy as np

# 환경 변수에서 카메라 비밀번호 로드
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")


# =============================================================================
# 공용 설정
# =============================================================================

# 공용 모델 경로 (NFS 공유 스토리지: /opt/airflow/data/checkpoints)
SHARED_MODEL_PATHS = {
    "yolo_detection": "/opt/airflow/data/checkpoints/objectdetection_yolo/best_250408.pt",
    "sam_segmentation": "/opt/airflow/data/checkpoints/segmentation_sam/sam_vit_h_4b8939.pth",
    "sam_yolo_cls": "/opt/airflow/data/checkpoints/segmentation_yolo_cls/best_250724.pt",
    "assembly_cls": "/opt/airflow/data/checkpoints/stage_cls/best_250312.pt",
}

# 공용 NFS 이미지 경로 (Bay별 이미지 저장 위치)
NFS_IMAGE_BASE_PATH = "/opt/airflow/data/NFS_images"


# =============================================================================
# Bay 설정
# =============================================================================

BAY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "12bay-west": {
        # === 기본 정보 ===
        "description": "12Bay Zone (12Bay west)",
        "enabled": True,

        # === 네트워크 설정 ===
        "network": {
            "camera_subnet": "10.150.160.0/24",
            "observer_ip": "10.150.160.1",
        },

        # === 스케줄링 설정 ===
        "schedule": {
            "observation": "*/10 * * * *",  # 10분마다
            "fusion": None,  # Trigger 기반
            "tracking": None,  # Trigger 기반
        },
        "operating_hours": [
            (0, 0, 23, 59),  # 24시간 운영
        ],
        "skip_times": [],  # 스킵 시간 없음

        # === 카메라 설정 ===
        "cameras": [
            {"name": "NS6130Y2509S001R1", "host": "10.150.160.183", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S002R1", "host": "10.150.160.184", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S003R1", "host": "10.150.160.193", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S004R1", "host": "10.150.160.194", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S005R1", "host": "10.150.160.209", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S006R1", "host": "10.150.160.211", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S007R1", "host": "10.150.160.221", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S008R1", "host": "10.150.160.222", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S009R1", "host": "10.150.160.224", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S010R1", "host": "10.150.160.225", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S011R1", "host": "10.150.160.226", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2509S012R1", "host": "10.150.160.227", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
        ],

        # === 카메라 외부 파라미터 (Extrinsic Parameters) ===
        "camera_extrinsics": {
            "NS6130Y2509S001R1": {"column": "C11", "coord": np.array([0, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2509S002R1": {"column": "C09", "coord": np.array([28, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2509S003R1": {"column": "C07", "coord": np.array([56, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2509S004R1": {"column": "C06", "coord": np.array([82, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2509S005R1": {"column": "C04", "coord": np.array([106, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2509S006R1": {"column": "C02", "coord": np.array([133, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2509S007R1": {"column": "D11", "coord": np.array([3, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2509S008R1": {"column": "D09", "coord": np.array([28, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2509S009R1": {"column": "D07", "coord": np.array([56, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2509S010R1": {"column": "D06", "coord": np.array([78, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2509S011R1": {"column": "D04", "coord": np.array([106, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2509S012R1": {"column": "D02", "coord": np.array([133, 0, 25], dtype=float), "pan": 0, "tilt": 50},
        },

        # === 이미지 획득 설정 ===
        # "file": NFS에서 이미지 파일 로드 (개발/백테스트용)
        # "onvif": 실시간 ONVIF 카메라 캡처 (운영용)
        "image_provider": "file",
        "image_provider_config": {
            # base_path는 bay_id로 동적 생성됨: get_image_provider_config(bay_id)
            "time_tolerance_hours": 100000.0,
        },

        # === 모델 설정 (NFS 공유 스토리지 사용) ===
        "models": SHARED_MODEL_PATHS.copy(),

        # === Factory Layout ===
        "factory_layout": {
            "width": 150,
            "height": 40,
            "vertical_center": 20,
            "areas": {
                # WEST SUBBAYS
                "A25": {"top_left": (2, 0), "bottom_right": (18, 20)},
                "A26": {"top_left": (18, 0), "bottom_right": (29, 20)},
                "A12": {"top_left": (29, 0), "bottom_right": (44, 20)},
                "A12_S": {"top_left": (29, 0), "bottom_right": (36.5, 20)},
                "A12_N": {"top_left": (36.5, 0), "bottom_right": (44, 20)},
                "A13": {"top_left": (44, 0), "bottom_right": (59, 20)},
                "A13_W": {"top_left": (44, 0), "bottom_right": (59, 10)},
                "A13_E": {"top_left": (44, 10), "bottom_right": (59, 20)},
                "A14": {"top_left": (59, 0), "bottom_right": (77, 20)},
                "A14_S": {"top_left": (59, 0), "bottom_right": (67, 20)},
                "A14_N": {"top_left": (67, 0), "bottom_right": (77, 20)},
                "-": {"top_left": (77, 0), "bottom_right": (86, 20)},
                "A1D": {"top_left": (86, 0), "bottom_right": (100, 20)},
                "A1E": {"top_left": (100, 0), "bottom_right": (115, 20)},
                "A1F": {"top_left": (115, 0), "bottom_right": (131, 20)},
                "A1G": {"top_left": (131, 0), "bottom_right": (147, 20)},
                # EAST SUBBAYS
                "A27": {"top_left": (2, 20), "bottom_right": (18, 40)},
                "A28": {"top_left": (18, 20), "bottom_right": (29, 40)},
                "A11": {"top_left": (29, 20), "bottom_right": (44, 40)},
                "A11_W": {"top_left": (29, 20), "bottom_right": (44, 30)},
                "A11_E": {"top_left": (29, 30), "bottom_right": (44, 40)},
                "A17": {"top_left": (44, 20), "bottom_right": (60, 40)},
                "A18(T/O)": {"top_left": (60, 20), "bottom_right": (82, 40)},
                "A19": {"top_left": (82, 20), "bottom_right": (97, 40)},
                "A1A": {"top_left": (97, 20), "bottom_right": (116, 40)},
                "A1B": {"top_left": (116, 20), "bottom_right": (131, 40)},
                "A1B_W": {"top_left": (116, 20), "bottom_right": (131, 30)},
                "A1B_E": {"top_left": (116, 30), "bottom_right": (131, 40)},
                "A1C": {"top_left": (131, 20), "bottom_right": (147, 40)},
            },
        },

        # === 추적 관련 설정 ===
        "tracking": {
            "iomin_threshold": 0.6,
            "contour_min_distance": 5.0,
            "exit_confirmation_threshold": 0.49,
            "hbeam_jig_subbays": ["A1D", "A1E", "A1F", "A1G"],
            "ignore_subbays": ["A18(T/O)", "-", "OB", "UNKNOWN"],
        },

        # === 투영 관련 설정 ===
        "projection": {
            "target_classes": ["panel", "block", "HP_block"],
            "pinjig_target_classes": ["pinjig", "hbeamjig", "gridjig", "pinjig_empty", "gridjig_empty", "hbeamjig_empty", "flatjig_empty"],
            "image_shape": (1080, 1920),
            "camera_intrinsics": {
                "fx": 2500.0,
                "fy": 2500.0,
                "cx": 960.0,
                "cy": 540.0,
            },
        },
    },

    "64bay": {
        # === 기본 정보 ===
        "description": "64Bay Zone",
        "enabled": True,

        # === 네트워크 설정 ===
        "network": {
            "camera_subnet": "10.150.161.0/24",
            "observer_ip": "10.150.161.1",
        },

        # === 스케줄링 설정 ===
        "schedule": {
            "observation": "*/10 * * * *",  # 10분마다
            "fusion": None,  # Trigger 기반
            "tracking": None,  # Trigger 기반
        },
        "operating_hours": [
            (0, 0, 23, 59),  # 24시간 운영
        ],
        "skip_times": [],  # 스킵 시간 없음

        # === 카메라 설정 (12bay와 동일한 카메라 구성) ===
        "cameras": [
            {"name": "NS6130Y2307S012", "host": "10.150.161.183", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S027", "host": "10.150.161.209", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S028", "host": "10.150.161.194", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S029", "host": "10.150.161.193", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S030", "host": "10.150.161.184", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S031", "host": "10.150.161.211", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S032", "host": "10.150.161.221", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S033", "host": "10.150.161.225", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S034", "host": "10.150.161.224", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S035", "host": "10.150.161.222", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S036", "host": "10.150.161.226", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
            {"name": "NS6130Y2310S037", "host": "10.150.161.227", "port": 80, "username": "admin", "password": CAMERA_PASSWORD},
        ],

        # === 카메라 외부 파라미터 (12bay와 동일) ===
        "camera_extrinsics": {
            "NS6130Y2307S012": {"column": "C11", "coord": np.array([0, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2310S027": {"column": "C09", "coord": np.array([28, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2310S028": {"column": "C07", "coord": np.array([56, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2310S029": {"column": "C06", "coord": np.array([82, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2310S030": {"column": "C04", "coord": np.array([106, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2310S031": {"column": "C02", "coord": np.array([133, 40, 25], dtype=float), "pan": 180, "tilt": 50},
            "NS6130Y2310S032": {"column": "D11", "coord": np.array([3, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2310S033": {"column": "D09", "coord": np.array([28, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2310S034": {"column": "D07", "coord": np.array([56, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2310S035": {"column": "D06", "coord": np.array([78, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2310S036": {"column": "D04", "coord": np.array([106, 0, 25], dtype=float), "pan": 0, "tilt": 50},
            "NS6130Y2310S037": {"column": "D02", "coord": np.array([133, 0, 25], dtype=float), "pan": 0, "tilt": 50},
        },

        # === 이미지 획득 설정 ===
        # "file": NFS에서 이미지 파일 로드 (개발/백테스트용)
        # "onvif": 실시간 ONVIF 카메라 캡처 (운영용)
        "image_provider": "file",
        "image_provider_config": {
            # base_path는 bay_id로 동적 생성됨: get_image_provider_config(bay_id)
            "time_tolerance_hours": 100000.0,
        },

        # === 모델 설정 (NFS 공유 스토리지 사용) ===
        "models": SHARED_MODEL_PATHS.copy(),

        # === Factory Layout (12bay와 동일) ===
        "factory_layout": {
            "width": 150,
            "height": 40,
            "vertical_center": 20,
            "areas": {
                # WEST SUBBAYS
                "A25": {"top_left": (2, 0), "bottom_right": (18, 20)},
                "A26": {"top_left": (18, 0), "bottom_right": (29, 20)},
                "A12": {"top_left": (29, 0), "bottom_right": (44, 20)},
                "A12_S": {"top_left": (29, 0), "bottom_right": (36.5, 20)},
                "A12_N": {"top_left": (36.5, 0), "bottom_right": (44, 20)},
                "A13": {"top_left": (44, 0), "bottom_right": (59, 20)},
                "A13_W": {"top_left": (44, 0), "bottom_right": (59, 10)},
                "A13_E": {"top_left": (44, 10), "bottom_right": (59, 20)},
                "A14": {"top_left": (59, 0), "bottom_right": (77, 20)},
                "A14_S": {"top_left": (59, 0), "bottom_right": (67, 20)},
                "A14_N": {"top_left": (67, 0), "bottom_right": (77, 20)},
                "-": {"top_left": (77, 0), "bottom_right": (86, 20)},
                "A1D": {"top_left": (86, 0), "bottom_right": (100, 20)},
                "A1E": {"top_left": (100, 0), "bottom_right": (115, 20)},
                "A1F": {"top_left": (115, 0), "bottom_right": (131, 20)},
                "A1G": {"top_left": (131, 0), "bottom_right": (147, 20)},
                # EAST SUBBAYS
                "A27": {"top_left": (2, 20), "bottom_right": (18, 40)},
                "A28": {"top_left": (18, 20), "bottom_right": (29, 40)},
                "A11": {"top_left": (29, 20), "bottom_right": (44, 40)},
                "A11_W": {"top_left": (29, 20), "bottom_right": (44, 30)},
                "A11_E": {"top_left": (29, 30), "bottom_right": (44, 40)},
                "A17": {"top_left": (44, 20), "bottom_right": (60, 40)},
                "A18(T/O)": {"top_left": (60, 20), "bottom_right": (82, 40)},
                "A19": {"top_left": (82, 20), "bottom_right": (97, 40)},
                "A1A": {"top_left": (97, 20), "bottom_right": (116, 40)},
                "A1B": {"top_left": (116, 20), "bottom_right": (131, 40)},
                "A1B_W": {"top_left": (116, 20), "bottom_right": (131, 30)},
                "A1B_E": {"top_left": (116, 30), "bottom_right": (131, 40)},
                "A1C": {"top_left": (131, 20), "bottom_right": (147, 40)},
            },
        },

        # === 추적 관련 설정 (12bay와 동일) ===
        "tracking": {
            "iomin_threshold": 0.6,
            "contour_min_distance": 5.0,
            "exit_confirmation_threshold": 0.49,
            "hbeam_jig_subbays": ["A1D", "A1E", "A1F", "A1G"],
            "ignore_subbays": ["A18(T/O)", "-", "OB", "UNKNOWN"],
        },

        # === 투영 관련 설정 (12bay와 동일) ===
        "projection": {
            "target_classes": ["panel", "block", "HP_block"],
            "pinjig_target_classes": ["pinjig", "hbeamjig", "gridjig", "pinjig_empty", "gridjig_empty", "hbeamjig_empty", "flatjig_empty"],
            "image_shape": (1080, 1920),
            "camera_intrinsics": {
                "fx": 2500.0,
                "fy": 2500.0,
                "cx": 960.0,
                "cy": 540.0,
            },
        },
    },
}


# =============================================================================
# 헬퍼 함수
# =============================================================================

def get_observer_queue(bay_id: str) -> str:
    """Observer Queue 이름 생성"""
    return f"obs_{bay_id}"


def get_processor_queue(bay_id: str) -> str:
    """Processor Queue 이름 생성"""
    return f"proc_{bay_id}"


def get_enabled_bays() -> Dict[str, Dict[str, Any]]:
    """활성화된 Bay만 반환"""
    return {
        bay_id: config
        for bay_id, config in BAY_CONFIGS.items()
        if config.get("enabled", False)
    }


def get_bay_config(bay_id: str) -> Dict[str, Any]:
    """특정 Bay의 설정 반환"""
    if bay_id not in BAY_CONFIGS:
        raise ValueError(f"Unknown bay_id: {bay_id}. Available: {list(BAY_CONFIGS.keys())}")
    return BAY_CONFIGS[bay_id]


def get_all_observer_queues() -> List[str]:
    """모든 활성화된 Bay의 Observer Queue 목록 반환"""
    return [get_observer_queue(bay_id) for bay_id in get_enabled_bays().keys()]


def get_all_processor_queues() -> List[str]:
    """모든 활성화된 Bay의 Processor Queue 목록 반환"""
    return [get_processor_queue(bay_id) for bay_id in get_enabled_bays().keys()]


def get_shared_model_paths() -> Dict[str, str]:
    """공용 모델 경로 반환"""
    return SHARED_MODEL_PATHS.copy()


def get_nfs_image_path(bay_id: str) -> str:
    """Bay별 NFS 이미지 경로 반환"""
    return f"{NFS_IMAGE_BASE_PATH}/{bay_id}"


def get_image_provider_config(bay_id: str) -> Dict[str, Any]:
    """Bay별 image_provider_config 반환 (base_path 동적 생성)"""
    config = get_bay_config(bay_id)
    provider_config = config.get("image_provider_config", {}).copy()
    provider_config["base_path"] = get_nfs_image_path(bay_id)
    return provider_config
