# -*- coding: utf-8 -*-
"""
Observation Logic Unit Tests

src/observation/logic.py의 순수 로직 함수들을 테스트합니다.
Airflow 의존성 없이 테스트 가능합니다.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
import numpy as np

from src.observation.logic import (
    CaptureResult,
    InferenceRunResult,
    capture_images,
    check_execution_conditions,
    generate_batch_id,
)
from src.observation.image_provider.base import CapturedImage, ImageProvider


class MockImageProvider(ImageProvider):
    """테스트용 Mock Image Provider"""

    def __init__(self, images_to_return: list = None, should_fail: bool = False):
        self.images_to_return = images_to_return or []
        self.should_fail = should_fail
        self.get_images_called = False
        self.last_call_args = None

    def get_images(self, reference_time, cameras, bay_id):
        self.get_images_called = True
        self.last_call_args = {
            'reference_time': reference_time,
            'cameras': cameras,
            'bay_id': bay_id,
        }
        if self.should_fail:
            raise RuntimeError("Mock provider failure")
        return self.images_to_return

    def validate_config(self, cameras):
        return True


class TestGenerateBatchId:
    """generate_batch_id 함수 테스트"""

    def test_generate_batch_id_format(self):
        """배치 ID 포맷 확인"""
        bay_id = "12bay-west"
        reference_time = datetime(2025, 6, 15, 10, 30, 45)

        batch_id = generate_batch_id(bay_id, reference_time)

        assert batch_id == "12bay-west_20250615-103045"

    def test_generate_batch_id_different_times(self):
        """다른 시간에 대해 다른 배치 ID 생성"""
        bay_id = "64bay"
        time1 = datetime(2025, 1, 1, 0, 0, 0)
        time2 = datetime(2025, 1, 1, 0, 0, 1)

        batch_id1 = generate_batch_id(bay_id, time1)
        batch_id2 = generate_batch_id(bay_id, time2)

        assert batch_id1 != batch_id2


class TestCaptureImages:
    """capture_images 함수 테스트"""

    def create_mock_captured_image(self, camera_name: str, bay_id: str) -> CapturedImage:
        """테스트용 CapturedImage 생성"""
        return CapturedImage(
            image_id=f"{bay_id}_{camera_name}_test",
            camera_name=camera_name,
            bay_id=bay_id,
            image_data=np.zeros((480, 640, 3), dtype=np.uint8),
            captured_at=datetime.now(),
            source_path=f"/tmp/test/{camera_name}.jpg",
            metadata={"test": True},
        )

    def test_capture_images_success(self):
        """정상 캡처 테스트"""
        bay_id = "12bay-west"
        cameras = [{"name": "cam1"}, {"name": "cam2"}]
        reference_time = datetime(2025, 6, 15, 10, 30, 0)

        mock_images = [
            self.create_mock_captured_image("cam1", bay_id),
            self.create_mock_captured_image("cam2", bay_id),
        ]
        provider = MockImageProvider(images_to_return=mock_images)

        result = capture_images(
            bay_id=bay_id,
            cameras=cameras,
            reference_time=reference_time,
            image_provider=provider,
        )

        assert result.success is True
        assert result.bay_id == bay_id
        assert result.captured_count == 2
        assert len(result.captured_images) == 2
        assert len(result.captured_images_metadata) == 2
        assert result.error_message is None

    def test_capture_images_empty(self):
        """이미지가 없는 경우 테스트"""
        bay_id = "64bay"
        cameras = [{"name": "cam1"}]
        reference_time = datetime.now()

        provider = MockImageProvider(images_to_return=[])

        result = capture_images(
            bay_id=bay_id,
            cameras=cameras,
            reference_time=reference_time,
            image_provider=provider,
        )

        assert result.success is True  # 빈 결과도 성공으로 처리
        assert result.captured_count == 0
        assert len(result.captured_images) == 0

    def test_capture_images_provider_failure(self):
        """Provider 실패 테스트"""
        bay_id = "12bay-west"
        cameras = [{"name": "cam1"}]
        reference_time = datetime.now()

        provider = MockImageProvider(should_fail=True)

        result = capture_images(
            bay_id=bay_id,
            cameras=cameras,
            reference_time=reference_time,
            image_provider=provider,
        )

        assert result.success is False
        assert result.captured_count == 0
        assert result.error_message is not None
        assert "Mock provider failure" in result.error_message

    def test_capture_images_metadata_format(self):
        """메타데이터 포맷 확인"""
        bay_id = "12bay-west"
        cameras = [{"name": "cam1"}]
        reference_time = datetime(2025, 6, 15, 10, 30, 0)

        mock_image = self.create_mock_captured_image("cam1", bay_id)
        provider = MockImageProvider(images_to_return=[mock_image])

        result = capture_images(
            bay_id=bay_id,
            cameras=cameras,
            reference_time=reference_time,
            image_provider=provider,
        )

        assert len(result.captured_images_metadata) == 1
        metadata = result.captured_images_metadata[0]

        # 필수 필드 확인
        assert 'image_id' in metadata
        assert 'camera_name' in metadata
        assert 'bay_id' in metadata
        assert 'source_path' in metadata
        assert 'captured_at' in metadata
        assert metadata['camera_name'] == "cam1"
        assert metadata['bay_id'] == bay_id

    def test_capture_images_calls_provider_correctly(self):
        """Provider가 올바른 인자로 호출되는지 확인"""
        bay_id = "64bay"
        cameras = [{"name": "cam1"}, {"name": "cam2"}]
        reference_time = datetime(2025, 6, 15, 10, 30, 0)

        provider = MockImageProvider(images_to_return=[])

        capture_images(
            bay_id=bay_id,
            cameras=cameras,
            reference_time=reference_time,
            image_provider=provider,
        )

        assert provider.get_images_called is True
        assert provider.last_call_args['bay_id'] == bay_id
        assert provider.last_call_args['cameras'] == cameras
        assert provider.last_call_args['reference_time'] == reference_time


class TestCheckExecutionConditions:
    """check_execution_conditions 함수 테스트"""

    def test_capture_failed_returns_false(self):
        """캡처 실패 시 False 반환"""
        result = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 10, 0, 0),
            operating_hours=[(0, 0, 23, 59)],
            skip_times=[],
            capture_success=False,
            captured_count=0,
            bay_id="test-bay",
        )
        assert result is False

    def test_no_images_captured_returns_false(self):
        """캡처된 이미지 없을 때 False 반환"""
        result = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 10, 0, 0),
            operating_hours=[(0, 0, 23, 59)],
            skip_times=[],
            capture_success=True,
            captured_count=0,
            bay_id="test-bay",
        )
        assert result is False

    def test_outside_operating_hours_returns_false(self):
        """운영 시간 외 False 반환"""
        result = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 23, 30, 0),  # 23:30
            operating_hours=[(8, 0, 18, 0)],  # 08:00 ~ 18:00
            skip_times=[],
            capture_success=True,
            captured_count=5,
            bay_id="test-bay",
        )
        assert result is False

    def test_inside_operating_hours_returns_true(self):
        """운영 시간 내 True 반환"""
        result = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 10, 0, 0),  # 10:00
            operating_hours=[(8, 0, 18, 0)],  # 08:00 ~ 18:00
            skip_times=[],
            capture_success=True,
            captured_count=5,
            bay_id="test-bay",
        )
        assert result is True

    def test_skip_time_returns_false(self):
        """스킵 시간대일 때 False 반환"""
        result = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 12, 0, 0),  # 12:00 (점심시간)
            operating_hours=[(0, 0, 23, 59)],
            skip_times=[(12, 0, 5)],  # 12:00 ±5분 스킵
            capture_success=True,
            captured_count=5,
            bay_id="test-bay",
        )
        assert result is False

    def test_multiple_operating_hours(self):
        """복수 운영 시간 테스트"""
        operating_hours = [
            (8, 0, 12, 0),   # 오전
            (13, 0, 18, 0),  # 오후
        ]

        # 오전 운영 시간 내
        result1 = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 10, 0, 0),
            operating_hours=operating_hours,
            skip_times=[],
            capture_success=True,
            captured_count=5,
            bay_id="test-bay",
        )
        assert result1 is True

        # 점심 시간 (운영 시간 외)
        result2 = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 12, 30, 0),
            operating_hours=operating_hours,
            skip_times=[],
            capture_success=True,
            captured_count=5,
            bay_id="test-bay",
        )
        assert result2 is False

        # 오후 운영 시간 내
        result3 = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 15, 0, 0),
            operating_hours=operating_hours,
            skip_times=[],
            capture_success=True,
            captured_count=5,
            bay_id="test-bay",
        )
        assert result3 is True

    def test_24_hour_operation(self):
        """24시간 운영 테스트"""
        result = check_execution_conditions(
            logical_date=datetime(2025, 6, 15, 3, 0, 0),  # 새벽 3시
            operating_hours=[(0, 0, 23, 59)],
            skip_times=[],
            capture_success=True,
            captured_count=5,
            bay_id="test-bay",
        )
        assert result is True


class TestCaptureResult:
    """CaptureResult 데이터클래스 테스트"""

    def test_capture_result_success(self):
        """성공 결과 생성"""
        result = CaptureResult(
            success=True,
            batch_id="test_20250615-100000",
            bay_id="12bay-west",
            captured_count=5,
        )

        assert result.success is True
        assert result.error_message is None

    def test_capture_result_failure(self):
        """실패 결과 생성"""
        result = CaptureResult(
            success=False,
            batch_id="test_20250615-100000",
            bay_id="12bay-west",
            captured_count=0,
            error_message="Connection timeout",
        )

        assert result.success is False
        assert result.error_message == "Connection timeout"


class TestInferenceRunResult:
    """InferenceRunResult 데이터클래스 테스트"""

    def test_inference_result_success(self):
        """성공 결과 생성"""
        result = InferenceRunResult(
            success=True,
            batch_id="test_20250615-100000",
            bay_id="12bay-west",
            result_path="/opt/airflow/data/observation/12bay-west/test/result.pkl",
            processed_count=5,
        )

        assert result.success is True
        assert result.error_message is None

    def test_inference_result_failure(self):
        """실패 결과 생성"""
        result = InferenceRunResult(
            success=False,
            batch_id="test_20250615-100000",
            bay_id="12bay-west",
            result_path="",
            processed_count=0,
            error_message="Model not found",
        )

        assert result.success is False
        assert result.error_message == "Model not found"
