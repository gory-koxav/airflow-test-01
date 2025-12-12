# -*- coding: utf-8 -*-
"""
Processing TaskFlow Tasks

TaskFlow API 기반 Task 래퍼 함수들입니다.
순수 비즈니스 로직(src/fusion/logic.py, src/tracking/logic.py)을 Airflow Task로 변환합니다.

Layer 구조:
1. Pure Logic (src/fusion/logic.py, src/tracking/logic.py) - Airflow 독립, 단위 테스트 가능
2. Task Wrapper (이 파일) - Airflow context 처리, Queue/Resource 설정
3. DAG Factory (factory.py) - DAG 조립 및 흐름 정의
"""

from airflow.sdk import task
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@task(task_id='process_fusion')
def process_fusion_task(
    bay_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fusion 처리 Task

    실행 위치: Main Server Host의 Processor

    다중 카메라 데이터를 융합하여 블록의 공간 정보를 분석합니다.
    """
    from airflow.sdk import get_current_context
    from src.fusion import run_fusion

    # dag_run.conf에서 batch_id 가져오기
    context = get_current_context()
    dag_run_conf = context['dag_run'].conf
    batch_id = dag_run_conf.get('batch_id')

    # 순수 로직 호출
    result = run_fusion(
        bay_id=bay_id,
        batch_id=batch_id,
        config=config,
    )

    return {
        'success': result.success,
        'batch_id': result.batch_id,
        'bay_id': result.bay_id,
        'result_path': result.result_path,
        'csv_path': result.csv_path,
        'merged_block_count': result.merged_block_count,
        'error_message': result.error_message,
    }


@task(task_id='process_tracking')
def process_tracking_task(
    bay_id: str,
) -> Dict[str, Any]:
    """
    Tracking 처리 Task

    실행 위치: Main Server Host의 Processor

    블록의 이동을 추적하고 이벤트를 기록합니다.
    """
    from airflow.sdk import get_current_context
    from src.tracking import run_tracking

    # dag_run.conf에서 batch_id 가져오기
    context = get_current_context()
    dag_run_conf = context['dag_run'].conf
    batch_id = dag_run_conf.get('batch_id')

    # 순수 로직 호출
    result = run_tracking(
        bay_id=bay_id,
        batch_id=batch_id,
    )

    return {
        'success': result.success,
        'batch_id': result.batch_id,
        'bay_id': result.bay_id,
        'matched_count': result.matched_count,
        'new_count': result.new_count,
        'active_trackers': result.active_trackers,
        'error_message': result.error_message,
    }
