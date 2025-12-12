# -*- coding: utf-8 -*-
"""
Tracking Logic Unit Tests

src/tracking/logic.py의 순수 로직 함수들을 테스트합니다.
Airflow 의존성 없이 테스트 가능합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.tracking.logic import (
    TrackingRunResult,
    run_tracking,
)


class TestTrackingRunResult:
    """TrackingRunResult 데이터클래스 테스트"""

    def test_tracking_result_success(self):
        """성공 결과 생성"""
        result = TrackingRunResult(
            success=True,
            batch_id="12bay-west_20250615-100000",
            bay_id="12bay-west",
            matched_count=3,
            new_count=2,
            active_trackers=5,
        )

        assert result.success is True
        assert result.matched_count == 3
        assert result.new_count == 2
        assert result.active_trackers == 5
        assert result.error_message is None

    def test_tracking_result_failure(self):
        """실패 결과 생성"""
        result = TrackingRunResult(
            success=False,
            batch_id="12bay-west_20250615-100000",
            bay_id="12bay-west",
            matched_count=0,
            new_count=0,
            active_trackers=0,
            error_message="Fusion data not found",
        )

        assert result.success is False
        assert result.error_message == "Fusion data not found"


class TestRunTracking:
    """run_tracking 함수 테스트"""

    @patch('src.tracking.logic.TrackingService')
    def test_run_tracking_success(self, mock_service_class):
        """정상 Tracking 실행 테스트"""
        # Mock 설정
        mock_service = MagicMock()
        mock_service.load_trackers_state.return_value = True
        mock_service.process_timestep.return_value = {
            'success': True,
            'matched_count': 3,
            'new_count': 1,
            'active_trackers': 4,
        }
        mock_service.save_trackers_state.return_value = None
        mock_service_class.return_value = mock_service

        # 테스트 실행
        result = run_tracking(
            bay_id="12bay-west",
            batch_id="12bay-west_20250615-100000",
        )

        # 검증
        assert result.success is True
        assert result.matched_count == 3
        assert result.new_count == 1
        assert result.active_trackers == 4
        mock_service.load_trackers_state.assert_called_once()
        mock_service.save_trackers_state.assert_called_once()

    @patch('src.tracking.logic.TrackingService')
    def test_run_tracking_exception(self, mock_service_class):
        """TrackingService 예외 처리 테스트"""
        mock_service = MagicMock()
        mock_service.load_trackers_state.side_effect = RuntimeError("State load failed")
        mock_service_class.return_value = mock_service

        result = run_tracking(
            bay_id="12bay-west",
            batch_id="12bay-west_20250615-100000",
        )

        assert result.success is False
        assert "State load failed" in result.error_message

    @patch('src.tracking.logic.TrackingService')
    def test_run_tracking_custom_thresholds(self, mock_service_class):
        """커스텀 threshold 테스트"""
        mock_service = MagicMock()
        mock_service.process_timestep.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        run_tracking(
            bay_id="64bay",
            batch_id="test",
            iou_threshold=0.5,
            existence_threshold=0.1,
        )

        # TrackingService 초기화 시 threshold 전달 확인
        call_kwargs = mock_service_class.call_args[1]
        assert call_kwargs['iou_threshold'] == 0.5
        assert call_kwargs['existence_threshold'] == 0.1

    @patch('src.tracking.logic.TrackingService')
    def test_run_tracking_default_thresholds(self, mock_service_class):
        """기본 threshold 테스트"""
        mock_service = MagicMock()
        mock_service.process_timestep.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        run_tracking(
            bay_id="12bay-west",
            batch_id="test",
        )

        # 기본값 확인
        call_kwargs = mock_service_class.call_args[1]
        assert call_kwargs['iou_threshold'] == 0.3
        assert call_kwargs['existence_threshold'] == 0.0

    @patch('src.tracking.logic.TrackingService')
    def test_run_tracking_process_timestep_called(self, mock_service_class):
        """process_timestep이 올바르게 호출되는지 테스트"""
        mock_service = MagicMock()
        mock_service.process_timestep.return_value = {
            'success': True,
            'matched_count': 0,
            'new_count': 0,
            'active_trackers': 0,
        }
        mock_service_class.return_value = mock_service

        batch_id = "12bay-west_20250615-100000"
        run_tracking(
            bay_id="12bay-west",
            batch_id=batch_id,
        )

        # process_timestep 호출 확인
        mock_service.process_timestep.assert_called_once()
        call_args = mock_service.process_timestep.call_args[0]
        assert call_args[1] == batch_id  # batch_id 확인
        assert isinstance(call_args[0], datetime)  # timestamp 확인

    @patch('src.tracking.logic.TrackingService')
    def test_run_tracking_state_persistence(self, mock_service_class):
        """상태 저장/로드 호출 순서 테스트"""
        mock_service = MagicMock()
        mock_service.process_timestep.return_value = {'success': True}
        mock_service_class.return_value = mock_service

        call_order = []
        mock_service.load_trackers_state.side_effect = lambda: call_order.append('load')
        mock_service.process_timestep.side_effect = lambda *args: (
            call_order.append('process'),
            {'success': True}
        )[1]
        mock_service.save_trackers_state.side_effect = lambda: call_order.append('save')

        run_tracking(
            bay_id="12bay-west",
            batch_id="test",
        )

        # 호출 순서 확인: load -> process -> save
        assert call_order == ['load', 'process', 'save']
