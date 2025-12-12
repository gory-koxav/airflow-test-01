# -*- coding: utf-8 -*-
"""
Maintenance Tasks

유지보수 관련 Task 함수들 (Layer 2: Task Wrapper)
"""

from airflow.sdk import task
from config.bay_configs import get_maintenance_config
from datetime import datetime, timedelta
import os
import glob
import shutil
import logging

logger = logging.getLogger(__name__)


@task(task_id='check_disk_space')
def check_disk_space() -> dict:
    """
    디스크 여유 공간 확인 (정보 수집용)

    공간이 부족하더라도 실패하지 않고 경고만 출력합니다.
    정리 작업은 공간이 부족할 때 더욱 필요하므로 계속 진행해야 합니다.

    운영 환경에서는 NFS 마운트된 파일시스템의 디스크 공간을 확인합니다.

    Returns:
        dict: 디스크 공간 정보 및 경고 여부
    """
    import subprocess

    config = get_maintenance_config()
    data_path = os.getenv('AIRFLOW_DATA_PATH', '/opt/airflow/data')

    # 마운트 정보 확인 (NFS 마운트 상태 검증)
    try:
        mount_info = subprocess.check_output(['df', '-h', data_path], text=True)
        logger.info(f"📂 Mount info for {data_path}:")
        logger.info(f"\n{mount_info}")
    except Exception as e:
        logger.warning(f"Could not get mount info: {e}")

    total, used, free = shutil.disk_usage(data_path)
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    usage_percent = (used / total) * 100

    min_required = config["safety"]["min_free_space_gb"]

    # 경고만 출력하고 계속 진행 (fail 시키지 않음)
    warning_issued = False
    if free_gb < min_required:
        logger.warning("⚠️  Low disk space detected!")
        logger.warning(f"Free space: {free_gb:.2f}GB (recommended minimum: {min_required}GB)")
        logger.warning("Cleanup is critically needed. Proceeding with cleanup tasks...")
        warning_issued = True
    else:
        logger.info(f"✓ Disk space OK: {free_gb:.2f}GB free ({usage_percent:.1f}% used)")

    return {
        "free_space_gb": round(free_gb, 2),
        "used_space_gb": round(used_gb, 2),
        "total_space_gb": round(total_gb, 2),
        "usage_percent": round(usage_percent, 2),
        "warning_issued": warning_issued,
        "below_minimum": free_gb < min_required,
    }


@task(task_id='cleanup_pkl_files')
def cleanup_old_pkl_files() -> dict:
    """
    오래된 .pkl 파일 삭제

    Returns:
        dict: 삭제된 파일 정보
    """
    config = get_maintenance_config()
    days_to_keep = config["data_retention"]["pkl_files"]
    dry_run = config["safety"]["dry_run"]

    cutoff_time = datetime.now() - timedelta(days=days_to_keep)
    data_path = os.getenv('AIRFLOW_DATA_PATH', '/opt/airflow/data')

    deleted_count = 0
    deleted_size_mb = 0
    scanned_count = 0

    for pkl_file in glob.glob(f"{data_path}/**/*.pkl", recursive=True):
        scanned_count += 1

        if os.path.getmtime(pkl_file) < cutoff_time.timestamp():
            file_size = os.path.getsize(pkl_file) / (1024 * 1024)  # MB

            if not dry_run:
                try:
                    os.remove(pkl_file)
                    deleted_count += 1
                    deleted_size_mb += file_size
                except Exception as e:
                    logger.error(f"Failed to delete {pkl_file}: {e}")
            else:
                logger.info(f"[DRY RUN] Would delete: {pkl_file} ({file_size:.2f}MB)")
                deleted_count += 1
                deleted_size_mb += file_size

    return {
        "scanned_files": scanned_count,
        "deleted_pkl_files": deleted_count,
        "freed_space_mb": round(deleted_size_mb, 2),
        "retention_days": days_to_keep,
        "dry_run": dry_run,
    }


@task(task_id='cleanup_json_files')
def cleanup_old_json_files() -> dict:
    """
    오래된 .json 파일 삭제

    Returns:
        dict: 삭제된 파일 정보
    """
    config = get_maintenance_config()
    days_to_keep = config["data_retention"]["json_files"]
    dry_run = config["safety"]["dry_run"]

    cutoff_time = datetime.now() - timedelta(days=days_to_keep)
    data_path = os.getenv('AIRFLOW_DATA_PATH', '/opt/airflow/data')

    deleted_count = 0
    deleted_size_mb = 0
    scanned_count = 0

    for json_file in glob.glob(f"{data_path}/**/*.json", recursive=True):
        scanned_count += 1

        if os.path.getmtime(json_file) < cutoff_time.timestamp():
            file_size = os.path.getsize(json_file) / (1024 * 1024)  # MB

            if not dry_run:
                try:
                    os.remove(json_file)
                    deleted_count += 1
                    deleted_size_mb += file_size
                except Exception as e:
                    logger.error(f"Failed to delete {json_file}: {e}")
            else:
                logger.info(f"[DRY RUN] Would delete: {json_file} ({file_size:.2f}MB)")
                deleted_count += 1
                deleted_size_mb += file_size

    return {
        "scanned_files": scanned_count,
        "deleted_json_files": deleted_count,
        "freed_space_mb": round(deleted_size_mb, 2),
        "retention_days": days_to_keep,
        "dry_run": dry_run,
    }


@task(task_id='cleanup_image_files')
def cleanup_old_image_files() -> dict:
    """
    오래된 이미지 파일 삭제 (.jpg, .png)

    Returns:
        dict: 삭제된 파일 정보
    """
    config = get_maintenance_config()
    days_to_keep = config["data_retention"]["image_files"]
    dry_run = config["safety"]["dry_run"]

    cutoff_time = datetime.now() - timedelta(days=days_to_keep)
    data_path = os.getenv('AIRFLOW_DATA_PATH', '/opt/airflow/data')

    deleted_count = 0
    deleted_size_mb = 0
    scanned_count = 0

    for pattern in ["**/*.jpg", "**/*.png", "**/*.jpeg"]:
        for img_file in glob.glob(f"{data_path}/{pattern}", recursive=True):
            scanned_count += 1

            if os.path.getmtime(img_file) < cutoff_time.timestamp():
                file_size = os.path.getsize(img_file) / (1024 * 1024)  # MB

                if not dry_run:
                    try:
                        os.remove(img_file)
                        deleted_count += 1
                        deleted_size_mb += file_size
                    except Exception as e:
                        logger.error(f"Failed to delete {img_file}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would delete: {img_file} ({file_size:.2f}MB)")
                    deleted_count += 1
                    deleted_size_mb += file_size

    return {
        "scanned_files": scanned_count,
        "deleted_image_files": deleted_count,
        "freed_space_mb": round(deleted_size_mb, 2),
        "retention_days": days_to_keep,
        "dry_run": dry_run,
    }


@task(task_id='cleanup_airflow_logs')
def cleanup_old_logs() -> dict:
    """
    오래된 Airflow 로그 파일 삭제

    Returns:
        dict: 삭제된 로그 정보
    """
    config = get_maintenance_config()
    days_to_keep = config["log_retention"]["airflow_task_logs"]
    dry_run = config["safety"]["dry_run"]

    cutoff_time = datetime.now() - timedelta(days=days_to_keep)
    logs_path = "/opt/airflow/logs"

    deleted_count = 0
    deleted_size_mb = 0
    scanned_count = 0

    for log_file in glob.glob(f"{logs_path}/**/*.log", recursive=True):
        scanned_count += 1

        if os.path.getmtime(log_file) < cutoff_time.timestamp():
            file_size = os.path.getsize(log_file) / (1024 * 1024)  # MB

            if not dry_run:
                try:
                    os.remove(log_file)
                    deleted_count += 1
                    deleted_size_mb += file_size
                except Exception as e:
                    logger.error(f"Failed to delete {log_file}: {e}")
            else:
                logger.info(f"[DRY RUN] Would delete: {log_file} ({file_size:.2f}MB)")
                deleted_count += 1
                deleted_size_mb += file_size

    return {
        "scanned_files": scanned_count,
        "deleted_log_files": deleted_count,
        "freed_space_mb": round(deleted_size_mb, 2),
        "retention_days": days_to_keep,
        "dry_run": dry_run,
    }


@task(task_id='cleanup_empty_directories')
def cleanup_empty_directories() -> dict:
    """
    빈 디렉토리 삭제

    Returns:
        dict: 삭제된 디렉토리 정보
    """
    config = get_maintenance_config()
    dry_run = config["safety"]["dry_run"]

    data_path = os.getenv('AIRFLOW_DATA_PATH', '/opt/airflow/data')

    deleted_count = 0

    # Bottom-up 방식으로 빈 디렉토리 찾기
    for root, dirs, files in os.walk(data_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)

            # 디렉토리가 비어있는지 확인
            if not os.listdir(dir_path):
                if not dry_run:
                    try:
                        os.rmdir(dir_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {dir_path}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would delete empty directory: {dir_path}")
                    deleted_count += 1

    return {
        "deleted_directories": deleted_count,
        "dry_run": dry_run,
    }
