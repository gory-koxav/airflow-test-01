# -*- coding: utf-8 -*-
"""
Log Cleanup DAG

오래된 로그 파일 정리를 위한 DAG (Layer 3: DAG Definition)
"""

from airflow import DAG
from datetime import datetime, timedelta
from config.settings import get_maintenance_config
from dags.maintenance.tasks import cleanup_old_logs


# =============================================================================
# DAG 정의: Log Cleanup
# =============================================================================

config = get_maintenance_config()

with DAG(
    dag_id="maintenance_log_cleanup",
    schedule=config["schedules"]["log_cleanup"],
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["maintenance", "cleanup", "logs"],
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # queue 지정 안 함 → default queue (Main 서버)
    },
    description="Clean up old Airflow log files",
) as dag:

    # 로그 정리
    log_cleanup = cleanup_old_logs()
