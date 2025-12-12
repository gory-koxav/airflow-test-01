# -*- coding: utf-8 -*-
"""
Observation Pure Logic Functions

Airflow 의존성이 없는 순수 비즈니스 로직 함수들입니다.
단위 테스트가 가능하며, DAG Task에서 래핑하여 사용합니다.

Usage:
    # 테스트에서 직접 호출 가능
    result = capture_images(bay_id, cameras, reference_time, image_provider)

    # DAG Task에서 래핑하여 사용
    @task
    def capture_task(...):
        return capture_images(...)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import traceback

from .image_provider.base import CapturedImage, ImageProvider
from .image_provider.factory import ImageProviderFactory
from .inference_service import InferenceService, InferenceResult
from .result_saver import ResultSaver, SaveResult

logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """이미지 캡처 결과"""
    success: bool
    batch_id: str
    bay_id: str
    captured_count: int
    captured_images: List[CapturedImage] = field(default_factory=list)
    captured_images_metadata: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class InferenceRunResult:
    """추론 실행 결과"""
    success: bool
    batch_id: str
    bay_id: str
    result_path: str
    processed_count: int
    error_message: Optional[str] = None


def generate_batch_id(bay_id: str, reference_time: datetime) -> str:
    """배치 ID 생성"""
    return f"{bay_id}_{reference_time.strftime('%Y%m%d-%H%M%S')}"


def capture_images(
    bay_id: str,
    cameras: List[Dict[str, Any]],
    reference_time: datetime,
    image_provider: ImageProvider,
) -> CaptureResult:
    """
    이미지 캡처 로직 (순수 함수)

    Args:
        bay_id: Bay 식별자
        cameras: 카메라 설정 리스트
        reference_time: 기준 시간 (Airflow logical_date 대신 명시적 파라미터)
        image_provider: 이미지 제공자 인스턴스

    Returns:
        CaptureResult: 캡처 결과 (Airflow XCom 대신 dataclass 반환)
    """
    batch_id = generate_batch_id(bay_id, reference_time)

    logger.info(f"[{bay_id}] ========== Starting image capture ==========")
    logger.info(f"[{bay_id}] Batch ID: {batch_id}")
    logger.info(f"[{bay_id}] Reference time: {reference_time}")
    logger.info(f"[{bay_id}] Number of cameras: {len(cameras) if cameras else 0}")

    if cameras:
        camera_names = [c.get('name', 'unknown') for c in cameras]
        logger.info(f"[{bay_id}] Camera names: {camera_names}")

    try:
        # 이미지 획득
        logger.info(f"[{bay_id}] Getting images from provider...")
        captured_images = image_provider.get_images(
            reference_time=reference_time,
            cameras=cameras,
            bay_id=bay_id,
        )

        logger.info(f"[{bay_id}] Captured {len(captured_images)} images")

        # 캡처된 이미지 상세 정보 로깅
        if captured_images:
            for img in captured_images:
                logger.info(
                    f"[{bay_id}] Captured image: camera={img.camera_name}, "
                    f"shape={img.image_data.shape if img.image_data is not None else 'None'}, "
                    f"source_path={img.source_path}, "
                    f"captured_at={img.captured_at}"
                )
        else:
            logger.warning(f"[{bay_id}] No images captured!")

        # 메타데이터 생성 (이미지 자체는 XCom으로 전달 불가)
        captured_images_metadata = [
            {
                'image_id': img.image_id,
                'camera_name': img.camera_name,
                'bay_id': img.bay_id,
                'source_path': img.source_path,
                'captured_at': img.captured_at.isoformat(),
                'metadata': img.metadata,
            }
            for img in captured_images
        ]

        logger.info(f"[{bay_id}] ========== Image capture completed ==========")

        return CaptureResult(
            success=True,
            batch_id=batch_id,
            bay_id=bay_id,
            captured_count=len(captured_images),
            captured_images=captured_images,
            captured_images_metadata=captured_images_metadata,
        )

    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"[{bay_id}] ========== Capture FAILED ==========")
        logger.error(f"[{bay_id}] Error: {error_msg}")
        logger.error(f"[{bay_id}] Stack trace:\n{stack_trace}")

        return CaptureResult(
            success=False,
            batch_id=batch_id,
            bay_id=bay_id,
            captured_count=0,
            error_message=error_msg,
        )


def check_execution_conditions(
    logical_date: datetime,
    operating_hours: List[tuple],
    skip_times: List[tuple],
    capture_success: bool,
    captured_count: int,
    bay_id: str,
) -> bool:
    """
    실행 조건 체크 로직 (순수 함수)

    Args:
        logical_date: 논리적 실행 시간
        operating_hours: 운영 시간 목록 [(start_h, start_m, end_h, end_m), ...]
        skip_times: 스킵 시간 목록 [(skip_h, skip_m, tolerance), ...]
        capture_success: 캡처 성공 여부
        captured_count: 캡처된 이미지 수
        bay_id: Bay 식별자 (로깅용)

    Returns:
        bool: True면 다음 단계 실행, False면 스킵
    """
    # 1. 캡처 결과 확인
    if not capture_success:
        logger.warning(f"[{bay_id}] Capture failed, skipping inference")
        return False

    if captured_count == 0:
        logger.warning(f"[{bay_id}] No images captured, skipping inference")
        return False

    # 2. 운영 시간 체크
    if not _is_in_operating_hours(logical_date, operating_hours):
        logger.info(f"[{bay_id}] Not in operating hours: {logical_date}")
        return False

    # 3. 스킵 시간 체크
    if _should_skip_execution(logical_date, skip_times):
        logger.info(f"[{bay_id}] In skip time window: {logical_date}")
        return False

    logger.info(f"[{bay_id}] Execution conditions met, proceeding with inference")
    return True


def _is_in_operating_hours(dt: datetime, operating_hours: List[tuple]) -> bool:
    """운영 시간 내인지 확인"""
    current_minutes = dt.hour * 60 + dt.minute
    for start_h, start_m, end_h, end_m in operating_hours:
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= current_minutes <= end_minutes:
            return True
    return False


def _should_skip_execution(dt: datetime, skip_times: List[tuple]) -> bool:
    """스킵 시간인지 확인"""
    current_minutes = dt.hour * 60 + dt.minute
    for skip_h, skip_m, tolerance in skip_times:
        skip_minutes = skip_h * 60 + skip_m
        if abs(current_minutes - skip_minutes) <= tolerance:
            return True
    return False


def run_inference(
    bay_id: str,
    batch_id: str,
    captured_images_metadata: List[Dict[str, Any]],
    models: Dict[str, str],
    base_path: str = "/opt/airflow/data",
) -> InferenceRunResult:
    """
    AI 추론 및 결과 저장 로직 (순수 함수)

    Args:
        bay_id: Bay 식별자
        batch_id: 배치 ID
        captured_images_metadata: 캡처된 이미지 메타데이터 리스트
        models: 모델 경로 딕셔너리
        base_path: 데이터 저장 기본 경로

    Returns:
        InferenceRunResult: 추론 결과
    """
    import cv2
    from datetime import datetime as dt
    from .image_provider.base import CapturedImage

    if not captured_images_metadata:
        logger.error(f"[{bay_id}] No captured images metadata available")
        return InferenceRunResult(
            success=False,
            batch_id=batch_id,
            bay_id=bay_id,
            result_path="",
            processed_count=0,
            error_message='No captured images metadata available',
        )

    # 메타데이터에서 이미지 파일을 로드하여 CapturedImage 객체 재구성
    captured_images = []
    for meta in captured_images_metadata:
        source_path = meta['source_path']
        image_data = cv2.imread(source_path)

        if image_data is None:
            logger.error(f"[{bay_id}] Failed to load image from {source_path}")
            continue

        captured_image = CapturedImage(
            image_id=meta['image_id'],
            camera_name=meta['camera_name'],
            bay_id=meta['bay_id'],
            image_data=image_data,
            captured_at=dt.fromisoformat(meta['captured_at']),
            source_path=source_path,
            metadata=meta.get('metadata', {}),
        )
        captured_images.append(captured_image)

    if not captured_images:
        logger.error(f"[{bay_id}] No images could be loaded from metadata")
        return InferenceRunResult(
            success=False,
            batch_id=batch_id,
            bay_id=bay_id,
            result_path="",
            processed_count=0,
            error_message='No images could be loaded from metadata',
        )

    logger.info(f"[{bay_id}] Loaded {len(captured_images)} images for inference")
    logger.info(f"[{bay_id}] Starting inference for batch {batch_id}")

    try:
        # 1. AI 추론
        inference_service = InferenceService(
            yolo_model_path=models.get('yolo_detection'),
            sam_model_path=models.get('sam_segmentation'),
            sam_yolo_cls_model_path=models.get('sam_yolo_cls'),
            assembly_cls_model_path=models.get('assembly_cls'),
        )
        inference_results = inference_service.run_inference(captured_images)

        # 2. 결과 저장 (공유 스토리지에 Pickle 저장)
        result_saver = ResultSaver(base_path=base_path)
        save_result = result_saver.save_results(
            captured_images=captured_images,
            inference_results=inference_results,
            batch_id=batch_id,
            bay_id=bay_id,
        )

        logger.info(f"[{bay_id}] Inference completed, results saved to {save_result.result_path}")

        return InferenceRunResult(
            success=save_result.success,
            batch_id=batch_id,
            bay_id=bay_id,
            result_path=save_result.result_path,
            processed_count=save_result.processed_count,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{bay_id}] Inference failed: {error_msg}")
        return InferenceRunResult(
            success=False,
            batch_id=batch_id,
            bay_id=bay_id,
            result_path="",
            processed_count=0,
            error_message=error_msg,
        )


def create_image_provider(
    provider_type: str,
    provider_config: Dict[str, Any],
) -> ImageProvider:
    """
    이미지 제공자 생성 헬퍼 함수

    Args:
        provider_type: 제공자 유형 ("file", "onvif")
        provider_config: 제공자 설정

    Returns:
        ImageProvider: 이미지 제공자 인스턴스
    """
    return ImageProviderFactory.create(provider_type, **provider_config)
