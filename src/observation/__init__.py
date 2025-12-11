# -*- coding: utf-8 -*-
"""
Observation 모듈

카메라 이미지 캡처 및 AI 추론을 담당합니다.

Components:
- image_provider: 이미지 획득 Strategy Pattern 구현
- inference_service: AI 추론 서비스 (YOLO, SAM, Classification)
- result_saver: 결과 저장 (Pickle 파일)
"""

from .image_provider import (
    ImageProvider,
    CapturedImage,
    OnvifImageProvider,
    FileImageProvider,
    ImageProviderFactory,
)

__all__ = [
    "ImageProvider",
    "CapturedImage",
    "OnvifImageProvider",
    "FileImageProvider",
    "ImageProviderFactory",
]
