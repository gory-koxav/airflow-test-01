# -*- coding: utf-8 -*-
"""
Fusion Logic Unit Tests

src/fusion/logic.py의 순수 로직 함수들을 테스트합니다.
Airflow 의존성 없이 테스트 가능합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.fusion.logic import (
    FusionRunResult,
    run_fusion,
)


class TestFusionRunResult:
    """FusionRunResult 데이터클래스 테스트"""

    def test_fusion_result_success(self):
        """성공 결과 생성"""
        result = FusionRunResult(
            success=True,
            batch_id="12bay-west_20250615-100000",
            bay_id="12bay-west",
            result_path="/opt/airflow/data/fusion/12bay-west/test/result.pkl",
            csv_path="/opt/airflow/data/fusion/12bay-west/test/result.csv",
            merged_block_count=5,
        )

        assert result.success is True
        assert result.merged_block_count == 5
        assert result.error_message is None

    def test_fusion_result_failure(self):
        """실패 결과 생성"""
        result = FusionRunResult(
            success=False,
            batch_id="12bay-west_20250615-100000",
            bay_id="12bay-west",
            result_path="",
            csv_path="",
            merged_block_count=0,
            error_message="Observation data not found",
        )

        assert result.success is False
        assert result.error_message == "Observation data not found"


class TestRunFusion:
    """run_fusion 함수 테스트"""

    @patch('src.fusion.logic.FusionService')
    @patch('src.fusion.logic.FusionResultSaver')
    def test_run_fusion_success(self, mock_saver_class, mock_service_class):
        """정상 Fusion 실행 테스트"""
        # Mock 설정
        mock_service = MagicMock()
        mock_service.process_batch.return_value = (
            [{"block": 1}, {"block": 2}],  # block_infos
            [{"merged": 1}],  # merged_block_infos
        )
        mock_service_class.return_value = mock_service

        mock_saver = MagicMock()
        mock_saver.save_results.return_value = MagicMock(
            success=True,
            pickle_path="/opt/airflow/data/fusion/test/result.pkl",
            csv_path="/opt/airflow/data/fusion/test/result.csv",
            merged_block_count=1,
        )
        mock_saver_class.return_value = mock_saver

        # 테스트 실행
        config = {
            "projection": {
                "camera_intrinsics": {"fx": 2500.0, "fy": 2500.0},
                "target_classes": ["block", "panel"],
            },
            "camera_extrinsics": {},
        }

        result = run_fusion(
            bay_id="12bay-west",
            batch_id="12bay-west_20250615-100000",
            config=config,
        )

        # 검증
        assert result.success is True
        assert result.merged_block_count == 1
        mock_service.process_batch.assert_called_once_with("12bay-west_20250615-100000")

    @patch('src.fusion.logic.FusionService')
    def test_run_fusion_service_exception(self, mock_service_class):
        """FusionService 예외 처리 테스트"""
        mock_service = MagicMock()
        mock_service.process_batch.side_effect = RuntimeError("Processing failed")
        mock_service_class.return_value = mock_service

        config = {
            "projection": {},
            "camera_extrinsics": {},
        }

        result = run_fusion(
            bay_id="12bay-west",
            batch_id="12bay-west_20250615-100000",
            config=config,
        )

        assert result.success is False
        assert "Processing failed" in result.error_message

    @patch('src.fusion.logic.FusionService')
    @patch('src.fusion.logic.FusionResultSaver')
    def test_run_fusion_uses_config_correctly(self, mock_saver_class, mock_service_class):
        """Config가 올바르게 전달되는지 테스트"""
        mock_service = MagicMock()
        mock_service.process_batch.return_value = ([], [])
        mock_service_class.return_value = mock_service

        mock_saver = MagicMock()
        mock_saver.save_results.return_value = MagicMock(
            success=True,
            pickle_path="",
            csv_path="",
            merged_block_count=0,
        )
        mock_saver_class.return_value = mock_saver

        config = {
            "projection": {
                "camera_intrinsics": {"fx": 3000.0, "fy": 3000.0, "cx": 960.0, "cy": 540.0},
                "target_classes": ["block"],
            },
            "camera_extrinsics": {"cam1": {"pan": 180}},
            "fusion_iou_threshold": 0.5,
        }

        run_fusion(
            bay_id="64bay",
            batch_id="64bay_20250615-100000",
            config=config,
        )

        # FusionService 초기화 시 올바른 인자 전달 확인
        mock_service_class.assert_called_once()
        call_kwargs = mock_service_class.call_args[1]
        assert call_kwargs['bay_id'] == "64bay"
        assert call_kwargs['iou_threshold'] == 0.5

    @patch('src.fusion.logic.FusionService')
    @patch('src.fusion.logic.FusionResultSaver')
    def test_run_fusion_default_iou_threshold(self, mock_saver_class, mock_service_class):
        """기본 IOU threshold 테스트"""
        mock_service = MagicMock()
        mock_service.process_batch.return_value = ([], [])
        mock_service_class.return_value = mock_service

        mock_saver = MagicMock()
        mock_saver.save_results.return_value = MagicMock(
            success=True,
            pickle_path="",
            csv_path="",
            merged_block_count=0,
        )
        mock_saver_class.return_value = mock_saver

        config = {
            "projection": {},
            "camera_extrinsics": {},
            # fusion_iou_threshold 미지정
        }

        run_fusion(
            bay_id="12bay-west",
            batch_id="test",
            config=config,
        )

        # 기본값 0.3 사용 확인
        call_kwargs = mock_service_class.call_args[1]
        assert call_kwargs['iou_threshold'] == 0.3
