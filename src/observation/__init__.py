# -*- coding: utf-8 -*-
"""
Observation 모듈

카메라 이미지 캡처 및 AI 추론을 담당합니다.

Components:
- image_provider: 이미지 획득 Strategy Pattern 구현
- inference: AI 추론 파이프라인 (Facade + Strategy Pattern)
    - InferencePipeline: 파이프라인 Facade
    - YOLOObjectDetector: 객체 탐지
    - SAMObjectBoundarySegmenter: 경계 분할
    - AutomaticSegmenter: 핀지그 자동 분할
    - MaskClassifier: 마스크 분류
- result_saver: 결과 저장 (Pickle 파일)
- logic: 순수 비즈니스 로직 함수 (Airflow 독립)
"""

from .image_provider import (
    ImageProvider,
    CapturedImage,
    OnvifImageProvider,
    FileImageProvider,
    ImageProviderFactory,
)
from .logic import (
    CaptureResult,
    InferenceRunResult,
    capture_images,
    check_execution_conditions,
    run_inference,
    create_image_provider,
    generate_batch_id,
)
from .inference import (
    InferencePipeline,
    InferenceStrategy,
    YOLOObjectDetector,
    SAMObjectBoundarySegmenter,
    AutomaticSegmenter,
    MaskClassifier,
)

__all__ = [
    # Image Provider
    "ImageProvider",
    "CapturedImage",
    "OnvifImageProvider",
    "FileImageProvider",
    "ImageProviderFactory",
    # Pure Logic Functions
    "CaptureResult",
    "InferenceRunResult",
    "capture_images",
    "check_execution_conditions",
    "run_inference",
    "create_image_provider",
    "generate_batch_id",
    # Inference Pipeline
    "InferencePipeline",
    "InferenceStrategy",
    "YOLOObjectDetector",
    "SAMObjectBoundarySegmenter",
    "AutomaticSegmenter",
    "MaskClassifier",
]
