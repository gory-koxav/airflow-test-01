# -*- coding: utf-8 -*-
"""
Image Provider 기본 클래스 및 데이터 모델

Strategy Pattern의 추상 베이스 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CapturedImage:
    """
    캡처된 이미지 데이터 모델

    Attributes:
        image_id: 고유 이미지 식별자 (bay_id + camera_name + timestamp)
        camera_name: 카메라 이름 (예: "C_2", "D_11")
        bay_id: Bay 식별자 (예: "12bay", "64bay")
        image_data: 이미지 데이터 (numpy array, BGR 형식)
        captured_at: 캡처 시간
        metadata: 추가 메타데이터 (선택적)
    """
    image_id: str
    camera_name: str
    bay_id: str
    image_data: np.ndarray
    captured_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """이미지 데이터 유효성 검사"""
        if self.image_data is None:
            raise ValueError("image_data cannot be None")
        if not isinstance(self.image_data, np.ndarray):
            raise TypeError("image_data must be a numpy array")

    @property
    def shape(self) -> tuple:
        """이미지 shape 반환"""
        return self.image_data.shape

    @property
    def is_valid(self) -> bool:
        """이미지가 유효한지 확인"""
        return (
            self.image_data is not None
            and len(self.image_data.shape) >= 2
            and self.image_data.shape[0] > 0
            and self.image_data.shape[1] > 0
        )


class ImageProvider(ABC):
    """
    이미지 획득을 위한 추상 베이스 클래스 (Strategy Pattern)

    모든 이미지 제공자는 이 인터페이스를 구현해야 합니다.
    """

    @abstractmethod
    def get_images(
        self,
        reference_time: datetime,
        cameras: List[Dict[str, Any]],
        bay_id: str,
    ) -> List[CapturedImage]:
        """
        지정된 시간과 카메라 설정에 따라 이미지를 획득합니다.

        Args:
            reference_time: 기준 시간 (캡처 시간 또는 백테스트 기준 시간)
            cameras: 카메라 설정 리스트
            bay_id: Bay 식별자

        Returns:
            List[CapturedImage]: 획득된 이미지 리스트
        """
        pass

    @abstractmethod
    def validate_config(self, cameras: List[Dict[str, Any]]) -> bool:
        """
        카메라 설정의 유효성을 검사합니다.

        Args:
            cameras: 카메라 설정 리스트

        Returns:
            bool: 유효성 검사 통과 여부
        """
        pass

    def get_single_image(
        self,
        reference_time: datetime,
        camera: Dict[str, Any],
        bay_id: str,
    ) -> Optional[CapturedImage]:
        """
        단일 카메라에서 이미지를 획득합니다.

        기본 구현은 get_images를 호출하여 첫 번째 결과를 반환합니다.
        서브클래스에서 최적화된 구현을 제공할 수 있습니다.

        Args:
            reference_time: 기준 시간
            camera: 카메라 설정
            bay_id: Bay 식별자

        Returns:
            Optional[CapturedImage]: 획득된 이미지 또는 None
        """
        images = self.get_images(reference_time, [camera], bay_id)
        return images[0] if images else None
