# -*- coding: utf-8 -*-
"""
Processing DAG Factory

Bay별 Fusion/Tracking DAG를 동적으로 생성합니다.

실행 위치: Main Server Host의 Processor
- Fusion DAG: process_fusion → trigger_tracking
- Tracking DAG: process_tracking

배포 독립성:
- 이 파일은 Processor 워커에만 배포됩니다.
- observation 관련 코드에 의존하지 않습니다.
"""

from airflow import DAG
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import logging

from config.settings import (
    BaySetting,
    get_enabled_bay_settings,
    get_processor_queue,
)
from dags.processing.tasks import (
    process_fusion_task,
    process_tracking_task,
)

logger = logging.getLogger(__name__)


def create_fusion_dag(bay_id: str, bay_setting: BaySetting) -> DAG:
    """
    Fusion DAG 생성 - 다중 카메라 융합 분석

    실행 위치: Main Server Host의 Processor

    Args:
        bay_id: Bay 식별자
        bay_setting: Bay 설정 (Pydantic 모델)

    Returns:
        DAG: Fusion DAG 인스턴스
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
        description=f'{bay_setting.deployment.description} - Multi-Camera Fusion',
        default_args=default_args,
        schedule=None,  # Trigger 기반
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['fusion', bay_id, 'processor'],
    )

    with dag:
        # Task 1: Fusion 처리
        fusion_result = process_fusion_task.override(queue=processor_queue)(
            bay_id=bay_id,
        )

        # Task 2: Tracking DAG 트리거 (Classic Operator 유지)
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

        # 의존성 설정
        fusion_result >> trigger_tracking

    return dag


def create_tracking_dag(bay_id: str, bay_setting: BaySetting) -> DAG:
    """
    Tracking DAG 생성 - 블록 이동 추적 및 기록

    실행 위치: Main Server Host의 Processor

    Args:
        bay_id: Bay 식별자
        bay_setting: Bay 설정 (Pydantic 모델)

    Returns:
        DAG: Tracking DAG 인스턴스
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
        description=f'{bay_setting.deployment.description} - Tracking',
        default_args=default_args,
        schedule=None,  # Trigger 기반
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['tracking', bay_id, 'processor'],
    )

    with dag:
        # Task 1: Tracking 처리
        process_tracking_task.override(queue=processor_queue)(
            bay_id=bay_id,
        )

    return dag


# =============================================================================
# DAG 동적 생성
# =============================================================================

# 활성화된 Bay에 대해 Fusion/Tracking DAG 생성
for bay_id, bay_setting in get_enabled_bay_settings().items():
    globals()[f"fusion_{bay_id}_dag"] = create_fusion_dag(bay_id, bay_setting)
    globals()[f"tracking_{bay_id}_dag"] = create_tracking_dag(bay_id, bay_setting)
