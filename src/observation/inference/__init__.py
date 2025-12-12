# -*- coding: utf-8 -*-
"""
AI Inference Module

YOLO Object Detection, SAM Segmentation, Classification 파이프라인을 제공합니다.
Legacy visionflow-legacy-observation의 inference 모듈을 Airflow 환경에 맞게 마이그레이션했습니다.
"""

from .facade import InferencePipeline
from .strategies import (
    InferenceStrategy,
    YOLOObjectDetector,
    SAMObjectBoundarySegmenter,
    AutomaticSegmenter,
    MaskClassifier,
)

__all__ = [
    "InferencePipeline",
    "InferenceStrategy",
    "YOLOObjectDetector",
    "SAMObjectBoundarySegmenter",
    "AutomaticSegmenter",
    "MaskClassifier",
]
