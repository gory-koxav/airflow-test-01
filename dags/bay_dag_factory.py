# -*- coding: utf-8 -*-
"""
Bay별 전체 파이프라인 DAG Factory

각 Bay에 대해 다음 DAG들을 동적으로 생성:
1. observation_{bay_id}_dag  → Observer에서 실행 (Bay 물리 서버)
2. fusion_{bay_id}_dag       → Processor에서 실행 (Main Server Host)
3. tracking_{bay_id}_dag     → Processor에서 실행 (Main Server Host)

명명 규칙:
- DAG: {task_type}_{bay_id}_dag (예: observation_12bay_dag)
- Queue:
  - Observer: obs_{bay_id} (예: obs_12bay)
  - Processor: proc_{bay_id} (예: proc_12bay)
"""

from airflow import DAG
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import logging

# 프로젝트 경로 추가
sys.path.insert(0, '/opt/airflow')

from config.bay_configs import (
    BAY_CONFIGS,
    get_observer_queue,
    get_processor_queue,
    get_enabled_bays,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 공통 헬퍼 함수
# =============================================================================

def is_in_operating_hours(dt: datetime, operating_hours: list) -> bool:
    """운영 시간 내인지 확인"""
    current_minutes = dt.hour * 60 + dt.minute
    for start_h, start_m, end_h, end_m in operating_hours:
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= current_minutes <= end_minutes:
            return True
    return False


def should_skip_execution(dt: datetime, skip_times: list) -> bool:
    """스킵 시간인지 확인"""
    current_minutes = dt.hour * 60 + dt.minute
    for skip_h, skip_m, tolerance in skip_times:
        skip_minutes = skip_h * 60 + skip_m
        if abs(current_minutes - skip_minutes) <= tolerance:
            return True
    return False


# =============================================================================
# Task 구현 함수
# =============================================================================

def run_capture_task(
    bay_id: str,
    cameras: list,
    image_provider_type: str,
    image_provider_config: dict,
    **context,
) -> dict:
    """
    이미지 캡처 태스크 실행

    실행 위치: Bay 물리 서버의 Observer
    """
    import traceback
    from datetime import datetime
    from src.observation.image_provider import ImageProviderFactory

    # logical_date가 None인 경우 (manual trigger) 현재 시간 사용
    logical_date = context.get('logical_date') or context.get('data_interval_end') or datetime.now()
    batch_id = f"{bay_id}_{logical_date.strftime('%Y%m%d-%H%M%S')}"

    logger.info(f"[{bay_id}] ========== Starting image capture ==========")
    logger.info(f"[{bay_id}] Batch ID: {batch_id}")
    logger.info(f"[{bay_id}] Logical date: {logical_date}")
    logger.info(f"[{bay_id}] Image provider type: {image_provider_type}")
    logger.info(f"[{bay_id}] Image provider config: {image_provider_config}")
    logger.info(f"[{bay_id}] Number of cameras: {len(cameras) if cameras else 0}")

    if cameras:
        camera_names = [c.get('name', 'unknown') for c in cameras]
        logger.info(f"[{bay_id}] Camera names: {camera_names}")

    try:
        # 1. Image Provider 생성
        logger.info(f"[{bay_id}] Creating ImageProvider...")
        image_provider = ImageProviderFactory.create(
            image_provider_type,
            **image_provider_config,
        )
        logger.info(f"[{bay_id}] ImageProvider created successfully: {type(image_provider).__name__}")

        # 2. 이미지 획득 (카메라 접근 - Observer에서만 가능)
        logger.info(f"[{bay_id}] Getting images from provider...")
        captured_images = image_provider.get_images(
            reference_time=logical_date,
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
                    f"captured_at={img.captured_at}"
                )
        else:
            logger.warning(f"[{bay_id}] No images captured! Check camera configuration and image paths.")

        # 3. 캡처 결과를 XCom에 저장 (다음 Task에서 사용)
        # Note: 이미지 데이터는 직렬화 문제로 XCom에 직접 저장하지 않음
        # 대신 captured_images 객체를 임시 저장하고 참조만 전달
        context['ti'].xcom_push(key='captured_images', value=captured_images)

        logger.info(f"[{bay_id}] ========== Image capture completed ==========")

        return {
            'success': True,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'captured_count': len(captured_images),
        }

    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"[{bay_id}] ========== Capture FAILED ==========")
        logger.error(f"[{bay_id}] Error: {error_msg}")
        logger.error(f"[{bay_id}] Stack trace:\n{stack_trace}")
        return {
            'success': False,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'error_message': error_msg,
        }


def check_execution_conditions(
    bay_id: str,
    operating_hours: list,
    skip_times: list,
    **context,
) -> bool:
    """
    수행 여부 판단 태스크

    실행 위치: Bay 물리 서버의 Observer

    판단 기준:
    - 운영 시간 내인지 확인
    - 스킵 시간대인지 확인
    - 캡처 성공 여부 확인

    Returns:
        bool: True면 다음 Task 실행, False면 downstream task 스킵
    """
    logical_date = context['logical_date']

    # 1. 캡처 결과 확인
    capture_result = context['ti'].xcom_pull(task_ids='capture_images')
    if not capture_result or not capture_result.get('success', False):
        logger.warning(f"[{bay_id}] Capture failed, skipping inference")
        return False

    if capture_result.get('captured_count', 0) == 0:
        logger.warning(f"[{bay_id}] No images captured, skipping inference")
        return False

    # 2. 운영 시간 체크
    if not is_in_operating_hours(logical_date, operating_hours):
        logger.info(f"[{bay_id}] Not in operating hours: {logical_date}")
        return False

    # 3. 스킵 시간 체크
    if should_skip_execution(logical_date, skip_times):
        logger.info(f"[{bay_id}] In skip time window: {logical_date}")
        return False

    logger.info(f"[{bay_id}] Execution conditions met, proceeding with inference")
    return True


def run_inference_task(bay_id: str, models: dict, **context) -> dict:
    """
    AI 추론 및 결과 저장 태스크

    실행 위치: Bay 물리 서버의 Observer
    """
    from src.observation.inference_service import InferenceService
    from src.observation.result_saver import ResultSaver

    # 캡처 Task에서 전달받은 데이터
    capture_result = context['ti'].xcom_pull(task_ids='capture_images')
    batch_id = capture_result['batch_id']

    # XCom에서 captured_images 가져오기
    captured_images = context['ti'].xcom_pull(task_ids='capture_images', key='captured_images')

    if not captured_images:
        logger.error(f"[{bay_id}] No captured images available")
        return {
            'success': False,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'error_message': 'No captured images available',
        }

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
        result_saver = ResultSaver()
        save_result = result_saver.save_results(
            captured_images=captured_images,
            inference_results=inference_results,
            batch_id=batch_id,
            bay_id=bay_id,
        )

        logger.info(f"[{bay_id}] Inference completed, results saved to {save_result.result_path}")

        return {
            'success': save_result.success,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'result_path': save_result.result_path,
            'processed_count': save_result.processed_count,
        }

    except Exception as e:
        logger.error(f"[{bay_id}] Inference failed: {e}")
        return {
            'success': False,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'error_message': str(e),
        }


def run_fusion_task(bay_id: str, config: dict, **context) -> dict:
    """
    Fusion 태스크 실행

    실행 위치: Main Server Host의 Processor

    다중 카메라 데이터를 융합하여 블록의 공간 정보를 분석합니다.
    Observation 결과 (Pickle)를 읽어서 처리하고 결과를 저장합니다.
    """
    from src.fusion import FusionService, FusionResultSaver

    dag_run_conf = context['dag_run'].conf
    batch_id = dag_run_conf.get('batch_id')

    logger.info(f"[{bay_id}] Starting fusion for batch {batch_id}")

    try:
        # 1. FusionService 초기화
        fusion_service = FusionService(
            bay_id=bay_id,
            camera_intrinsics=config.get("camera_intrinsics", {}),
            camera_extrinsics=config.get("camera_extrinsics", {}),
            boundary_target_classes=config.get("boundary_target_classes", ["block", "panel"]),
            base_path="/opt/airflow/data",
            iou_threshold=config.get("fusion_iou_threshold", 0.3),
        )

        # 2. 배치 처리
        block_infos, merged_block_infos = fusion_service.process_batch(batch_id)

        # 3. 결과 저장
        result_saver = FusionResultSaver(base_path="/opt/airflow/data")
        save_result = result_saver.save_results(
            merged_block_infos=merged_block_infos,
            batch_id=batch_id,
            bay_id=bay_id,
        )

        logger.info(
            f"[{bay_id}] Fusion completed for batch {batch_id}: "
            f"{len(block_infos)} blocks -> {len(merged_block_infos)} merged blocks"
        )

        return {
            'success': save_result.success,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'result_path': save_result.pickle_path,
            'csv_path': save_result.csv_path,
            'merged_block_count': save_result.merged_block_count,
        }

    except Exception as e:
        logger.error(f"[{bay_id}] Fusion failed: {e}")
        return {
            'success': False,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'error_message': str(e),
        }


def run_tracking_task(bay_id: str, **context) -> dict:
    """
    Tracking 태스크 실행

    실행 위치: Main Server Host의 Processor

    블록의 이동을 추적하고 이벤트를 기록합니다.
    Fusion 결과 (Pickle/CSV)를 읽어서 처리합니다.
    """
    from src.tracking import TrackingService
    from datetime import datetime

    dag_run_conf = context['dag_run'].conf
    batch_id = dag_run_conf.get('batch_id')

    logger.info(f"[{bay_id}] Starting tracking for batch {batch_id}")

    try:
        # 1. TrackingService 초기화
        tracking_service = TrackingService(
            bay_id=bay_id,
            base_path="/opt/airflow/data",
            iou_threshold=0.3,
            existence_threshold=0.0,
        )

        # 2. 이전 상태 로드 (있으면)
        tracking_service.load_trackers_state()

        # 3. Timestep 처리
        timestamp = datetime.now()
        result = tracking_service.process_timestep(timestamp, batch_id)

        # 4. 현재 상태 저장
        tracking_service.save_trackers_state()

        logger.info(
            f"[{bay_id}] Tracking completed for batch {batch_id}: "
            f"matched={result.get('matched_count', 0)}, "
            f"new={result.get('new_count', 0)}, "
            f"active={result.get('active_trackers', 0)}"
        )

        return {
            'success': result.get('success', False),
            'batch_id': batch_id,
            'bay_id': bay_id,
            'matched_count': result.get('matched_count', 0),
            'new_count': result.get('new_count', 0),
            'active_trackers': result.get('active_trackers', 0),
        }

    except Exception as e:
        logger.error(f"[{bay_id}] Tracking failed: {e}")
        return {
            'success': False,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'error_message': str(e),
        }


# =============================================================================
# DAG Factory 함수
# =============================================================================

def create_observation_dag(bay_id: str, config: Dict[str, Any]) -> DAG:
    """
    Observation DAG 생성 - 카메라 캡처 및 AI 추론

    실행 위치: Bay별 물리 서버의 Observer
    """
    dag_id = f"observation_{bay_id}_dag"
    observer_queue = get_observer_queue(bay_id)

    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    }

    dag = DAG(
        dag_id=dag_id,
        description=f'{config["description"]} - CCTV Observation',
        default_args=default_args,
        schedule=config["schedule"]["observation"],
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['observation', bay_id, 'cctv', 'observer'],
    )

    with dag:
        # Task 1: 이미지 캡처 (Bay 물리 서버의 Observer에서 실행)
        capture_images = PythonOperator(
            task_id='capture_images',
            python_callable=run_capture_task,
            op_kwargs={
                "bay_id": bay_id,
                "cameras": config["cameras"],
                "image_provider_type": config["image_provider"],
                "image_provider_config": config.get("image_provider_config", {}),
            },
            queue=observer_queue,
        )

        # Task 2: 수행 여부 판단 (Bay 물리 서버의 Observer에서 실행)
        check_should_proceed = ShortCircuitOperator(
            task_id='check_should_proceed',
            python_callable=check_execution_conditions,
            op_kwargs={
                "bay_id": bay_id,
                "operating_hours": config["operating_hours"],
                "skip_times": config["skip_times"],
            },
            queue=observer_queue,
        )

        # Task 3: AI 추론 및 결과 저장 (Bay 물리 서버의 Observer에서 실행)
        run_inference = PythonOperator(
            task_id='run_inference',
            python_callable=run_inference_task,
            op_kwargs={
                "bay_id": bay_id,
                "models": config.get("models", {}),
            },
            queue=observer_queue,
        )

        # Task 4: Fusion DAG 트리거 (Bay 물리 서버의 Observer에서 실행)
        trigger_fusion = TriggerDagRunOperator(
            task_id='trigger_fusion',
            trigger_dag_id=f'fusion_{bay_id}_dag',
            conf={
                'batch_id': '{{ ti.xcom_pull(task_ids="run_inference")["batch_id"] }}',
                'bay_id': bay_id,
                'observation_result_path': '{{ ti.xcom_pull(task_ids="run_inference")["result_path"] }}',
            },
            wait_for_completion=False,
            queue=observer_queue,
        )

        capture_images >> check_should_proceed >> run_inference >> trigger_fusion

    return dag


def create_fusion_dag(bay_id: str, config: Dict[str, Any]) -> DAG:
    """
    Fusion DAG 생성 - 다중 카메라 융합 분석

    실행 위치: Main Server Host의 Processor
    """
    dag_id = f"fusion_{bay_id}_dag"
    processor_queue = get_processor_queue(bay_id)

    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    }

    dag = DAG(
        dag_id=dag_id,
        description=f'{config["description"]} - Multi-Camera Fusion',
        default_args=default_args,
        schedule=None,  # Trigger 기반
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['fusion', bay_id, 'processor'],
    )

    with dag:
        # Main Server Host의 Processor에서 실행
        process_fusion = PythonOperator(
            task_id='process_fusion',
            python_callable=run_fusion_task,
            op_kwargs={
                "bay_id": bay_id,
                "config": config,
            },
            queue=processor_queue,
        )

        trigger_tracking = TriggerDagRunOperator(
            task_id='trigger_tracking',
            trigger_dag_id=f'tracking_{bay_id}_dag',
            conf={
                'batch_id': '{{ dag_run.conf["batch_id"] }}',
                'bay_id': bay_id,
                'fusion_result_path': '{{ ti.xcom_pull(task_ids="process_fusion")["result_path"] }}',
            },
            wait_for_completion=False,
            queue=processor_queue,
        )

        process_fusion >> trigger_tracking

    return dag


def create_tracking_dag(bay_id: str, config: Dict[str, Any]) -> DAG:
    """
    Tracking DAG 생성 - 블록 이동 추적 및 기록

    실행 위치: Main Server Host의 Processor
    """
    dag_id = f"tracking_{bay_id}_dag"
    processor_queue = get_processor_queue(bay_id)

    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    }

    dag = DAG(
        dag_id=dag_id,
        description=f'{config["description"]} - Tracking',
        default_args=default_args,
        schedule=None,  # Trigger 기반
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['tracking', bay_id, 'processor'],
    )

    with dag:
        # Main Server Host의 Processor에서 실행
        process_tracking = PythonOperator(
            task_id='process_tracking',
            python_callable=run_tracking_task,
            op_kwargs={"bay_id": bay_id},
            queue=processor_queue,
        )

    return dag


# =============================================================================
# DAG 동적 생성
# =============================================================================

# 활성화된 Bay에 대해 DAG 생성
for bay_id, config in get_enabled_bays().items():
    # 각 Bay에 대해 3개의 DAG 생성
    globals()[f"observation_{bay_id}_dag"] = create_observation_dag(bay_id, config)
    globals()[f"fusion_{bay_id}_dag"] = create_fusion_dag(bay_id, config)
    globals()[f"tracking_{bay_id}_dag"] = create_tracking_dag(bay_id, config)
