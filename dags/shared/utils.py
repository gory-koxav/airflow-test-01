# -*- coding: utf-8 -*-
"""
Shared Utility Functions

DAG 간 공유되는 유틸리티 함수들입니다.
"""

from datetime import datetime
from typing import List


def is_in_operating_hours(dt: datetime, operating_hours: List[tuple]) -> bool:
    """
    운영 시간 내인지 확인

    Args:
        dt: 확인할 시간
        operating_hours: 운영 시간 목록 [(start_h, start_m, end_h, end_m), ...]

    Returns:
        bool: 운영 시간 내이면 True
    """
    current_minutes = dt.hour * 60 + dt.minute
    for start_h, start_m, end_h, end_m in operating_hours:
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= current_minutes <= end_minutes:
            return True
    return False


def should_skip_execution(dt: datetime, skip_times: List[tuple]) -> bool:
    """
    스킵 시간인지 확인

    Args:
        dt: 확인할 시간
        skip_times: 스킵 시간 목록 [(skip_h, skip_m, tolerance), ...]

    Returns:
        bool: 스킵 시간이면 True
    """
    current_minutes = dt.hour * 60 + dt.minute
    for skip_h, skip_m, tolerance in skip_times:
        skip_minutes = skip_h * 60 + skip_m
        if abs(current_minutes - skip_minutes) <= tolerance:
            return True
    return False
