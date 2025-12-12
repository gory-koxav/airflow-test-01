# -*- coding: utf-8 -*-
"""
Observation TaskFlow Tasks

TaskFlow API 기반 Task 래퍼 함수들입니다.
순수 비즈니스 로직(src/observation/logic.py)을 Airflow Task로 변환합니다.

Layer 구조:
1. Pure Logic (src/observation/logic.py) - Airflow 독립, 단위 테스트 가능
2. Task Wrapper (이 파일) - Airflow context 처리, Queue/Resource 설정
3. DAG Factory (factory.py) - DAG 조립 및 흐름 정의
"""

from airflow.sdk import task
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@task(task_id='capture_images')
def capture_images_task(
    bay_id: str,
    cameras: List[Dict[str, Any]],
    image_provider_type: str,
    image_provider_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    이미지 캡처 Task

    실행 위치: Bay 물리 서버의 Observer
    """
    from airflow.sdk import get_current_context
    from src.observation import capture_images, create_image_provider

    # Airflow context에서 logical_date 가져오기
    context = get_current_context()
    logical_date = context.get('logical_date') or context.get('data_interval_end') or datetime.now()

    # 이미지 제공자 생성
    image_provider = create_image_provider(image_provider_type, image_provider_config)

    # 순수 로직 호출
    result = capture_images(
        bay_id=bay_id,
        cameras=cameras,
        reference_time=logical_date,
        image_provider=image_provider,
    )

    # TaskFlow는 return으로 XCom 전달 (ti.xcom_push 불필요)
    return {
        'success': result.success,
        'batch_id': result.batch_id,
        'bay_id': result.bay_id,
        'captured_count': result.captured_count,
        'captured_images_metadata': result.captured_images_metadata,
        'error_message': result.error_message,
    }


@task.short_circuit(task_id='check_should_proceed')
def check_should_proceed_task(
    capture_result: Dict[str, Any],
    bay_id: str,
    operating_hours: List[tuple],
    skip_times: List[tuple],
) -> bool:
    """
    수행 여부 판단 Task (ShortCircuit)

    실행 위치: Bay 물리 서버의 Observer

    Returns:
        bool: True면 다음 Task 실행, False면 downstream task 스킵
    """
    from airflow.sdk import get_current_context
    from src.observation import check_execution_conditions

    context = get_current_context()
    logical_date = context.get('logical_date') or datetime.now()

    # 순수 로직 호출
    return check_execution_conditions(
        logical_date=logical_date,
        operating_hours=operating_hours,
        skip_times=skip_times,
        capture_success=capture_result.get('success', False),
        captured_count=capture_result.get('captured_count', 0),
        bay_id=bay_id,
    )


@task(task_id='run_inference')
def run_inference_task(
    capture_result: Dict[str, Any],
    bay_id: str,
) -> Dict[str, Any]:
    """
    AI 추론 및 결과 저장 Task

    실행 위치: Bay 물리 서버의 Observer

    InferencePipeline (Facade)을 사용하여 전체 AI 추론 파이프라인을 실행합니다.
    파이프라인 순서 (config/settings.py에서 설정):
    1. YOLOObjectDetector - 객체 탐지
    2. AutomaticSegmenter - 핀지그 자동 분할
    3. MaskClassifier - 분할된 마스크 분류
    4. SAMObjectBoundarySegmenter - 객체 경계 분할
    """
    from src.observation import run_inference

    batch_id = capture_result['batch_id']
    captured_images_metadata = capture_result.get('captured_images_metadata', [])

    # 순수 로직 호출 (pipeline_config=None이면 config.settings에서 자동 로드)
    result = run_inference(
        bay_id=bay_id,
        batch_id=batch_id,
        captured_images_metadata=captured_images_metadata,
    )

    return {
        'success': result.success,
        'batch_id': result.batch_id,
        'bay_id': result.bay_id,
        'result_path': result.result_path,
        'processed_count': result.processed_count,
        'error_message': result.error_message,
    }
