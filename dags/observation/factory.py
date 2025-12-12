# -*- coding: utf-8 -*-
"""
Observation DAG Factory

Bay별 Observation DAG를 동적으로 생성합니다.

실행 위치: Bay 물리 서버의 Observer
파이프라인: capture_images → check_should_proceed → run_inference → trigger_fusion

배포 독립성:
- 이 파일은 Observer 워커에만 배포됩니다.
- fusion/tracking 관련 코드에 의존하지 않습니다.
"""

from airflow import DAG
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from config.bay_configs import (
    get_enabled_bays,
    get_observer_queue,
    get_image_provider_config,
)
from dags.observation.tasks import (
    capture_images_task,
    check_should_proceed_task,
    run_inference_task,
)

logger = logging.getLogger(__name__)


def create_observation_dag(bay_id: str, config: Dict[str, Any]) -> DAG:
    """
    Observation DAG 생성 - 카메라 캡처 및 AI 추론

    실행 위치: Bay별 물리 서버의 Observer

    Args:
        bay_id: Bay 식별자
        config: Bay 설정

    Returns:
        DAG: Observation DAG 인스턴스
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
        # Task 1: 이미지 캡처
        # TaskFlow API: .override()로 queue 설정
        capture_result = capture_images_task.override(queue=observer_queue)(
            bay_id=bay_id,
            cameras=config["cameras"],
            image_provider_type=config["image_provider"],
            image_provider_config=get_image_provider_config(bay_id),
        )

        # Task 2: 수행 여부 판단 (ShortCircuit)
        should_proceed = check_should_proceed_task.override(queue=observer_queue)(
            capture_result=capture_result,
            bay_id=bay_id,
            operating_hours=config["operating_hours"],
            skip_times=config["skip_times"],
        )

        # Task 3: AI 추론 및 결과 저장
        inference_result = run_inference_task.override(queue=observer_queue)(
            capture_result=capture_result,
            bay_id=bay_id,
            models=config.get("models", {}),
        )

        # Task 4: Fusion DAG 트리거 (Classic Operator 유지)
        # TriggerDagRunOperator는 오케스트레이션 기능이므로 TaskFlow로 변환하지 않음
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

        # 의존성 설정
        # TaskFlow에서는 함수 호출 순서로 암시적 의존성이 생기지만,
        # ShortCircuit과 Classic Operator 연결은 명시적으로 설정
        capture_result >> should_proceed >> inference_result >> trigger_fusion

    return dag


# =============================================================================
# DAG 동적 생성
# =============================================================================

# 활성화된 Bay에 대해 Observation DAG 생성
for bay_id, config in get_enabled_bays().items():
    globals()[f"observation_{bay_id}_dag"] = create_observation_dag(bay_id, config)
