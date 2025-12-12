# -*- coding: utf-8 -*-
"""
Fusion 모듈

다중 카메라 데이터를 융합하여 블록의 공간 정보를 분석합니다.

Components:
- FusionService: 메인 Fusion 처리 서비스
- Projector: 카메라 좌표 -> 공장 좌표 변환
- OverlapAnalyzer: 마스크/박스 중첩 분석
- models: Fusion 관련 데이터 모델
- logic: 순수 비즈니스 로직 함수 (Airflow 독립)
"""

from .models import FrameData, ProjectedData, MergedBlockInfo
from .projector import Projector
from .overlap_analyzer import OverlapAnalyzer
from .fusion_service import FusionService
from .result_saver import FusionResultSaver
from .logic import FusionRunResult, run_fusion

__all__ = [
    # Models & Services
    "FrameData",
    "ProjectedData",
    "MergedBlockInfo",
    "Projector",
    "OverlapAnalyzer",
    "FusionService",
    "FusionResultSaver",
    # Pure Logic Functions
    "FusionRunResult",
    "run_fusion",
]
