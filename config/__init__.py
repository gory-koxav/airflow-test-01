# -*- coding: utf-8 -*-
"""
Configuration 모듈

Bay별 설정 및 공통 설정을 관리합니다.
"""

from .bay_configs import (
    BAY_CONFIGS,
    get_observer_queue,
    get_processor_queue,
    get_enabled_bays,
    get_bay_config,
)

__all__ = [
    "BAY_CONFIGS",
    "get_observer_queue",
    "get_processor_queue",
    "get_enabled_bays",
    "get_bay_config",
]
