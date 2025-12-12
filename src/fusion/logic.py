# -*- coding: utf-8 -*-
"""
Fusion Pure Logic Functions

Airflow 의존성이 없는 순수 비즈니스 로직 함수들입니다.
단위 테스트가 가능하며, DAG Task에서 래핑하여 사용합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from .fusion_service import FusionService
from .result_saver import FusionResultSaver

logger = logging.getLogger(__name__)


@dataclass
class FusionRunResult:
    """Fusion 실행 결과"""
    success: bool
    batch_id: str
    bay_id: str
    result_path: str
    csv_path: str
    merged_block_count: int
    error_message: Optional[str] = None


def run_fusion(
    bay_id: str,
    batch_id: str,
    config: Dict[str, Any],
    base_path: str = "/opt/airflow/data",
) -> FusionRunResult:
    """
    Fusion 로직 (순수 함수)

    다중 카메라 데이터를 융합하여 블록의 공간 정보를 분석합니다.
    Observation 결과 (Pickle)를 읽어서 처리하고 결과를 저장합니다.

    Args:
        bay_id: Bay 식별자
        batch_id: 배치 ID
        config: Bay 설정 (camera_intrinsics, camera_extrinsics 등)
        base_path: 데이터 저장 기본 경로

    Returns:
        FusionRunResult: Fusion 실행 결과
    """
    logger.info(f"[{bay_id}] Starting fusion for batch {batch_id}")

    try:
        # 1. FusionService 초기화
        fusion_service = FusionService(
            bay_id=bay_id,
            camera_intrinsics=config.get("projection", {}).get("camera_intrinsics", {}),
            camera_extrinsics=config.get("camera_extrinsics", {}),
            boundary_target_classes=config.get("projection", {}).get(
                "target_classes", ["block", "panel"]
            ),
            base_path=base_path,
            iou_threshold=config.get("fusion_iou_threshold", 0.3),
        )

        # 2. 배치 처리
        block_infos, merged_block_infos = fusion_service.process_batch(batch_id)

        # 3. 결과 저장
        result_saver = FusionResultSaver(base_path=base_path)
        save_result = result_saver.save_results(
            merged_block_infos=merged_block_infos,
            batch_id=batch_id,
            bay_id=bay_id,
        )

        logger.info(
            f"[{bay_id}] Fusion completed for batch {batch_id}: "
            f"{len(block_infos)} blocks -> {len(merged_block_infos)} merged blocks"
        )

        return FusionRunResult(
            success=save_result.success,
            batch_id=batch_id,
            bay_id=bay_id,
            result_path=save_result.pickle_path,
            csv_path=save_result.csv_path,
            merged_block_count=save_result.merged_block_count,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{bay_id}] Fusion failed: {error_msg}")
        return FusionRunResult(
            success=False,
            batch_id=batch_id,
            bay_id=bay_id,
            result_path="",
            csv_path="",
            merged_block_count=0,
            error_message=error_msg,
        )
