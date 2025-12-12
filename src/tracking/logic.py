# -*- coding: utf-8 -*-
"""
Tracking Pure Logic Functions

Airflow 의존성이 없는 순수 비즈니스 로직 함수들입니다.
단위 테스트가 가능하며, DAG Task에서 래핑하여 사용합니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from .tracking_service import TrackingService

logger = logging.getLogger(__name__)


@dataclass
class TrackingRunResult:
    """Tracking 실행 결과"""
    success: bool
    batch_id: str
    bay_id: str
    matched_count: int
    new_count: int
    active_trackers: int
    error_message: Optional[str] = None


def run_tracking(
    bay_id: str,
    batch_id: str,
    base_path: str = "/opt/airflow/data",
    iou_threshold: float = 0.3,
    existence_threshold: float = 0.0,
) -> TrackingRunResult:
    """
    Tracking 로직 (순수 함수)

    블록의 이동을 추적하고 이벤트를 기록합니다.
    Fusion 결과 (Pickle/CSV)를 읽어서 처리합니다.

    Args:
        bay_id: Bay 식별자
        batch_id: 배치 ID
        base_path: 데이터 저장 기본 경로
        iou_threshold: IOU 매칭 임계값
        existence_threshold: 존재 확률 임계값

    Returns:
        TrackingRunResult: Tracking 실행 결과
    """
    logger.info(f"[{bay_id}] Starting tracking for batch {batch_id}")

    try:
        # 1. TrackingService 초기화
        tracking_service = TrackingService(
            bay_id=bay_id,
            base_path=base_path,
            iou_threshold=iou_threshold,
            existence_threshold=existence_threshold,
        )

        # 2. 이전 상태 로드 (있으면)
        tracking_service.load_trackers_state()

        # 3. Timestep 처리
        timestamp = datetime.now()
        result = tracking_service.process_timestep(timestamp, batch_id)

        # 처리 실패 시 예외 발생 (Airflow Task를 실패로 처리하기 위함)
        if not result.get('success', False):
            error_msg = result.get('error_message', 'Unknown error during tracking')
            raise RuntimeError(f"Failed to process tracking: {error_msg}")

        # 4. 현재 상태 저장
        tracking_service.save_trackers_state()

        logger.info(
            f"[{bay_id}] Tracking completed for batch {batch_id}: "
            f"matched={result.get('matched_count', 0)}, "
            f"new={result.get('new_count', 0)}, "
            f"active={result.get('active_trackers', 0)}"
        )

        return TrackingRunResult(
            success=True,
            batch_id=batch_id,
            bay_id=bay_id,
            matched_count=result.get('matched_count', 0),
            new_count=result.get('new_count', 0),
            active_trackers=result.get('active_trackers', 0),
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{bay_id}] Tracking failed: {error_msg}")
        # 예외를 다시 발생시켜 Airflow Task를 실패로 처리
        raise
