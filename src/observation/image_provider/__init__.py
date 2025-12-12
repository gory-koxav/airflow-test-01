# -*- coding: utf-8 -*-
"""
Image Provider 모듈

Strategy Pattern을 사용하여 다양한 이미지 획득 방식을 지원합니다:
- OnvifImageProvider: ONVIF 프로토콜을 통한 실시간 카메라 캡처
- FileImageProvider: 파일 시스템에서 이미지 로드 (백테스트용)

CapturedImage는 src/observation/models.py에 정의되어 있으며,
하위 호환성을 위해 여기서도 re-export합니다.
"""

from .base import ImageProvider, CapturedImage
from .onvif_provider import OnvifImageProvider
from .file_provider import FileImageProvider
from .factory import ImageProviderFactory

__all__ = [
    "ImageProvider",
    "CapturedImage",
    "OnvifImageProvider",
    "FileImageProvider",
    "ImageProviderFactory",
]
