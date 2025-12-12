# -*- coding: utf-8 -*-
"""
Tracking 모듈

블록의 이동을 추적하고 이벤트를 기록합니다.

Components:
- TrackingService: 메인 Tracking 처리 서비스
- BlockTracker: 개별 블록 상태 관리
- EventLogger: 이벤트 기록
- models: Tracking 관련 데이터 모델
- logic: 순수 비즈니스 로직 함수 (Airflow 독립)
"""

from .models import (
    BlockState,
    EventType,
    AssembleState,
    BlockId,
    BBox,
    TrackingEvent,
    CSVDetection,
)
from .block_tracker import BlockTracker
from .event_logger import EventLogger
from .tracking_service import TrackingService
from .logic import TrackingRunResult, run_tracking

__all__ = [
    # Models
    "BlockState",
    "EventType",
    "AssembleState",
    "BlockId",
    "BBox",
    "TrackingEvent",
    "CSVDetection",
    # Services
    "BlockTracker",
    "EventLogger",
    "TrackingService",
    # Pure Logic Functions
    "TrackingRunResult",
    "run_tracking",
]
