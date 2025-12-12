# -*- coding: utf-8 -*-
"""
Data Cleanup DAG

오래된 데이터 파일 정리를 위한 DAG (Layer 3: DAG Definition)
"""

from airflow import DAG
from datetime import datetime, timedelta
from config.bay_configs import get_maintenance_config
from dags.maintenance.tasks import (
    check_disk_space,
    cleanup_old_pkl_files,
    cleanup_old_json_files,
    cleanup_old_image_files,
    cleanup_empty_directories,
)


# =============================================================================
# DAG 정의: Data Cleanup
# =============================================================================

config = get_maintenance_config()

with DAG(
    dag_id="maintenance_data_cleanup",
    schedule=config["schedules"]["data_cleanup"],
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["maintenance", "cleanup", "data"],
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # queue 지정 안 함 → default queue (Main 서버)
    },
    description="Clean up old data files (pkl, json, images) from shared storage",
) as dag:

    # 1. 디스크 공간 체크
    disk_check = check_disk_space()

    # 2. 파일 정리 (병렬 실행)
    pkl_cleanup = cleanup_old_pkl_files()
    json_cleanup = cleanup_old_json_files()
    image_cleanup = cleanup_old_image_files()

    # 3. 빈 디렉토리 정리
    dir_cleanup = cleanup_empty_directories()

    # 의존성: 디스크 체크 → 파일 정리 → 디렉토리 정리
    disk_check >> [pkl_cleanup, json_cleanup, image_cleanup] >> dir_cleanup
