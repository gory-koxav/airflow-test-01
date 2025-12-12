# -*- coding: utf-8 -*-
"""
Inference Strategy Base Class

AI 모델 추론 전략에 대한 추상 인터페이스를 정의합니다.
모든 추론 전략 클래스는 이 인터페이스를 구현해야 합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.observation.models import CapturedImage


class InferenceStrategy(ABC):
    """
    AI 모델 추론 전략 인터페이스 (Strategy Pattern)

    각 전략 클래스는 이 인터페이스를 구현하여 특정 AI 모델 추론을 수행합니다.
    FacadeInferencePipeline에서 순차적으로 호출되며, inference_results 딕셔너리를
    누적하여 다음 전략으로 전달합니다.
    """

    @abstractmethod
    def run(
        self,
        captured_images: List[CapturedImage],
        inference_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        캡처된 이미지 리스트를 받아 추론을 수행하고,
        결과 딕셔너리를 업데이트하여 반환합니다.

        Args:
            captured_images: 캡처된 이미지 리스트
            inference_results: 이전 전략들의 추론 결과가 누적된 딕셔너리
                - 키: image_id
                - 값: 해당 이미지의 추론 결과 (detections, boundary_masks 등)

        Returns:
            Dict[str, Any]: 업데이트된 추론 결과 딕셔너리
        """
        pass
