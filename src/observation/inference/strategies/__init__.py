# -*- coding: utf-8 -*-
"""
Inference Strategies Module

AI 모델 추론을 위한 Strategy 패턴 구현입니다.
각 전략 클래스는 InferenceStrategy 인터페이스를 구현합니다.
"""

from .base import InferenceStrategy
from .object_detector import YOLOObjectDetector
from .boundary_segmenter import SAMObjectBoundarySegmenter
from .automatic_segmenter import AutomaticSegmenter
from .mask_classifier import MaskClassifier

__all__ = [
    "InferenceStrategy",
    "YOLOObjectDetector",
    "SAMObjectBoundarySegmenter",
    "AutomaticSegmenter",
    "MaskClassifier",
]
