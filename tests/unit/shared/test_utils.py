# -*- coding: utf-8 -*-
"""
Shared Utils Unit Tests

dags/shared/utils.py의 유틸리티 함수들을 테스트합니다.
"""

import pytest
from datetime import datetime

from dags.shared.utils import (
    is_in_operating_hours,
    should_skip_execution,
)


class TestIsInOperatingHours:
    """is_in_operating_hours 함수 테스트"""

    def test_inside_single_range(self):
        """단일 운영 시간 범위 내"""
        dt = datetime(2025, 6, 15, 10, 30, 0)  # 10:30
        operating_hours = [(8, 0, 18, 0)]  # 08:00 ~ 18:00

        assert is_in_operating_hours(dt, operating_hours) is True

    def test_outside_single_range(self):
        """단일 운영 시간 범위 외"""
        dt = datetime(2025, 6, 15, 20, 0, 0)  # 20:00
        operating_hours = [(8, 0, 18, 0)]  # 08:00 ~ 18:00

        assert is_in_operating_hours(dt, operating_hours) is False

    def test_at_start_boundary(self):
        """시작 경계"""
        dt = datetime(2025, 6, 15, 8, 0, 0)  # 08:00
        operating_hours = [(8, 0, 18, 0)]  # 08:00 ~ 18:00

        assert is_in_operating_hours(dt, operating_hours) is True

    def test_at_end_boundary(self):
        """종료 경계"""
        dt = datetime(2025, 6, 15, 18, 0, 0)  # 18:00
        operating_hours = [(8, 0, 18, 0)]  # 08:00 ~ 18:00

        assert is_in_operating_hours(dt, operating_hours) is True

    def test_just_before_start(self):
        """시작 직전"""
        dt = datetime(2025, 6, 15, 7, 59, 0)  # 07:59
        operating_hours = [(8, 0, 18, 0)]  # 08:00 ~ 18:00

        assert is_in_operating_hours(dt, operating_hours) is False

    def test_just_after_end(self):
        """종료 직후"""
        dt = datetime(2025, 6, 15, 18, 1, 0)  # 18:01
        operating_hours = [(8, 0, 18, 0)]  # 08:00 ~ 18:00

        assert is_in_operating_hours(dt, operating_hours) is False

    def test_multiple_ranges_first(self):
        """복수 운영 시간 - 첫 번째 범위"""
        dt = datetime(2025, 6, 15, 10, 0, 0)  # 10:00
        operating_hours = [
            (8, 0, 12, 0),   # 08:00 ~ 12:00
            (13, 0, 18, 0),  # 13:00 ~ 18:00
        ]

        assert is_in_operating_hours(dt, operating_hours) is True

    def test_multiple_ranges_second(self):
        """복수 운영 시간 - 두 번째 범위"""
        dt = datetime(2025, 6, 15, 15, 0, 0)  # 15:00
        operating_hours = [
            (8, 0, 12, 0),   # 08:00 ~ 12:00
            (13, 0, 18, 0),  # 13:00 ~ 18:00
        ]

        assert is_in_operating_hours(dt, operating_hours) is True

    def test_multiple_ranges_gap(self):
        """복수 운영 시간 - 사이 간격"""
        dt = datetime(2025, 6, 15, 12, 30, 0)  # 12:30 (점심시간)
        operating_hours = [
            (8, 0, 12, 0),   # 08:00 ~ 12:00
            (13, 0, 18, 0),  # 13:00 ~ 18:00
        ]

        assert is_in_operating_hours(dt, operating_hours) is False

    def test_24_hour_operation(self):
        """24시간 운영"""
        operating_hours = [(0, 0, 23, 59)]

        # 자정
        assert is_in_operating_hours(datetime(2025, 6, 15, 0, 0, 0), operating_hours) is True
        # 새벽
        assert is_in_operating_hours(datetime(2025, 6, 15, 3, 30, 0), operating_hours) is True
        # 낮
        assert is_in_operating_hours(datetime(2025, 6, 15, 12, 0, 0), operating_hours) is True
        # 밤
        assert is_in_operating_hours(datetime(2025, 6, 15, 23, 59, 0), operating_hours) is True

    def test_empty_operating_hours(self):
        """빈 운영 시간"""
        dt = datetime(2025, 6, 15, 10, 0, 0)
        operating_hours = []

        assert is_in_operating_hours(dt, operating_hours) is False

    def test_with_minutes(self):
        """분 단위 경계"""
        dt = datetime(2025, 6, 15, 9, 30, 0)  # 09:30
        operating_hours = [(9, 30, 17, 30)]  # 09:30 ~ 17:30

        assert is_in_operating_hours(dt, operating_hours) is True


class TestShouldSkipExecution:
    """should_skip_execution 함수 테스트"""

    def test_at_skip_time(self):
        """스킵 시간 정확히"""
        dt = datetime(2025, 6, 15, 12, 0, 0)  # 12:00
        skip_times = [(12, 0, 5)]  # 12:00 ±5분

        assert should_skip_execution(dt, skip_times) is True

    def test_within_tolerance(self):
        """tolerance 범위 내"""
        dt = datetime(2025, 6, 15, 12, 3, 0)  # 12:03
        skip_times = [(12, 0, 5)]  # 12:00 ±5분

        assert should_skip_execution(dt, skip_times) is True

    def test_at_tolerance_boundary(self):
        """tolerance 경계"""
        dt = datetime(2025, 6, 15, 12, 5, 0)  # 12:05
        skip_times = [(12, 0, 5)]  # 12:00 ±5분

        assert should_skip_execution(dt, skip_times) is True

    def test_outside_tolerance(self):
        """tolerance 범위 외"""
        dt = datetime(2025, 6, 15, 12, 10, 0)  # 12:10
        skip_times = [(12, 0, 5)]  # 12:00 ±5분

        assert should_skip_execution(dt, skip_times) is False

    def test_before_skip_time(self):
        """스킵 시간 이전 (tolerance 내)"""
        dt = datetime(2025, 6, 15, 11, 57, 0)  # 11:57
        skip_times = [(12, 0, 5)]  # 12:00 ±5분

        assert should_skip_execution(dt, skip_times) is True

    def test_multiple_skip_times(self):
        """복수 스킵 시간"""
        skip_times = [
            (12, 0, 5),   # 점심
            (18, 0, 5),   # 퇴근
        ]

        # 점심 시간
        assert should_skip_execution(datetime(2025, 6, 15, 12, 0, 0), skip_times) is True
        # 퇴근 시간
        assert should_skip_execution(datetime(2025, 6, 15, 18, 0, 0), skip_times) is True
        # 일반 시간
        assert should_skip_execution(datetime(2025, 6, 15, 15, 0, 0), skip_times) is False

    def test_empty_skip_times(self):
        """빈 스킵 시간"""
        dt = datetime(2025, 6, 15, 12, 0, 0)
        skip_times = []

        assert should_skip_execution(dt, skip_times) is False

    def test_zero_tolerance(self):
        """tolerance가 0인 경우"""
        skip_times = [(12, 0, 0)]

        # 정확히 12:00
        assert should_skip_execution(datetime(2025, 6, 15, 12, 0, 0), skip_times) is True
        # 12:01
        assert should_skip_execution(datetime(2025, 6, 15, 12, 1, 0), skip_times) is False

    def test_large_tolerance(self):
        """큰 tolerance"""
        dt = datetime(2025, 6, 15, 12, 25, 0)  # 12:25
        skip_times = [(12, 0, 30)]  # 12:00 ±30분

        assert should_skip_execution(dt, skip_times) is True
