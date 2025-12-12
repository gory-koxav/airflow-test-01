# -*- coding: utf-8 -*-
"""
Pytest Configuration

테스트 환경 설정 및 공통 fixture를 정의합니다.
"""

import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 PYTHONPATH에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def sample_bay_config():
    """테스트용 Bay 설정"""
    return {
        "description": "Test Bay",
        "enabled": True,
        "schedule": {
            "observation": "*/10 * * * *",
            "fusion": None,
            "tracking": None,
        },
        "operating_hours": [(0, 0, 23, 59)],
        "skip_times": [],
        "cameras": [
            {"name": "cam1", "host": "192.168.1.1", "port": 80},
            {"name": "cam2", "host": "192.168.1.2", "port": 80},
        ],
        "camera_extrinsics": {},
        "image_provider": "file",
        "image_provider_config": {
            "base_path": "/tmp/test",
            "time_tolerance_hours": 1.0,
        },
        "models": {
            "yolo_detection": "/path/to/yolo.pt",
            "sam_segmentation": "/path/to/sam.pth",
        },
        "projection": {
            "camera_intrinsics": {
                "fx": 2500.0,
                "fy": 2500.0,
                "cx": 960.0,
                "cy": 540.0,
            },
            "target_classes": ["block", "panel"],
        },
    }


@pytest.fixture
def sample_cameras():
    """테스트용 카메라 목록"""
    return [
        {"name": "cam1", "host": "192.168.1.1", "port": 80, "username": "admin", "password": "test"},
        {"name": "cam2", "host": "192.168.1.2", "port": 80, "username": "admin", "password": "test"},
    ]


@pytest.fixture
def sample_operating_hours():
    """테스트용 운영 시간"""
    return [
        (8, 0, 12, 0),   # 오전
        (13, 0, 18, 0),  # 오후
    ]


@pytest.fixture
def sample_skip_times():
    """테스트용 스킵 시간"""
    return [
        (12, 0, 5),   # 점심시간 ±5분
        (18, 0, 5),   # 퇴근시간 ±5분
    ]
