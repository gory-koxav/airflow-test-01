# -*- coding: utf-8 -*-
"""
Inference Pipeline Facade

AI 추론 파이프라인을 조립하고 실행하는 Facade 패턴 구현입니다.
Legacy visionflow-legacy-observation의 facade.py를 마이그레이션했습니다.

설정 파일(config/settings.py)의 pipeline 설정에 따라 전략 클래스를 동적으로 로드하고
순차적으로 실행합니다.
"""

import importlib
import logging
from typing import Dict, Any, List, Optional

from src.observation.models import CapturedImage
from .strategies.base import InferenceStrategy

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    AI 추론 파이프라인 Facade

    설정에 정의된 전략 클래스들을 순차적으로 실행하여
    이미지에 대한 전체 AI 추론을 수행합니다.

    Pipeline 순서 (기본 설정):
    1. YOLOObjectDetector - 객체 탐지
    2. AutomaticSegmenter - 핀지그 자동 분할
    3. MaskClassifier - 분할된 마스크 분류
    4. SAMObjectBoundarySegmenter - 객체 경계 분할

    Attributes:
        _pipeline: 전략 인스턴스 딕셔너리 (이름 -> 인스턴스)
    """

    def __init__(self, pipeline_config: List[Dict[str, Any]] = None):
        """
        Args:
            pipeline_config: 파이프라인 설정 리스트
                None이면 config.settings에서 기본 설정 로드

                예시:
                [
                    {"name": "YOLOObjectDetector", "module": "src.observation.inference.strategies", "args": {}},
                    {"name": "AutomaticSegmenter", "module": "src.observation.inference.strategies", "args": {}},
                    ...
                ]
        """
        self._pipeline: Dict[str, InferenceStrategy] = {}

        if pipeline_config is None:
            from config.settings import get_setting
            setting = get_setting()
            pipeline_config = [item.model_dump() for item in setting.model.pipeline]

        self._load_pipeline(pipeline_config)

    def _load_pipeline(self, pipeline_config: List[Dict[str, Any]]) -> None:
        """
        파이프라인 전략 클래스 동적 로드

        Args:
            pipeline_config: 파이프라인 설정 리스트
        """
        logger.info("Loading inference pipeline...")

        for model_spec in pipeline_config:
            model_name = model_spec.get("name")
            model_path = model_spec.get("module")
            model_args = model_spec.get("args", {})

            if not (model_name and model_path):
                continue

            try:
                module = importlib.import_module(model_path)
                model_class = getattr(module, model_name)
                instance = model_class(**model_args)
                self._pipeline[model_name] = instance
                logger.info(f"Loaded strategy: {model_name} from {model_path}")

            except ImportError as e:
                logger.error(f"Failed to import module '{model_path}': {e}")
            except AttributeError as e:
                logger.error(f"Failed to find class '{model_name}' in '{model_path}': {e}")
            except TypeError as e:
                logger.error(
                    f"Failed to initialize '{model_name}'. "
                    f"Check args in config: {e}"
                )

        logger.info(f"Pipeline loaded with {len(self._pipeline)} strategies")

    def process_batch(
        self,
        captured_images: List[CapturedImage],
    ) -> Dict[str, Any]:
        """
        이미지 배치에 대해 전체 AI 추론 파이프라인 실행

        Args:
            captured_images: 캡처된 이미지 리스트

        Returns:
            Dict[str, Any]: 추론 결과 딕셔너리
                - 키: image_id
                - 값: 해당 이미지의 추론 결과
                    - detections: 객체 탐지 결과
                    - boundary_masks: 경계 마스크
                    - pinjig_masks: 핀지그 마스크
                    - pinjig_classifications: 핀지그 분류 결과
        """
        logger.info("--- AI Inference Pipeline: Starting ---")

        inference_results: Dict[str, Any] = {}

        for key, strategy in self._pipeline.items():
            logger.info(f"Running strategy: {key}")
            try:
                inference_results = strategy.run(captured_images, inference_results)
            except Exception as e:
                logger.error(f"Strategy '{key}' failed: {e}")
                raise

        logger.info("--- AI Inference Pipeline: Finished ---")
        return inference_results

    def get_strategy(self, name: str) -> Optional[InferenceStrategy]:
        """
        이름으로 전략 인스턴스 조회

        Args:
            name: 전략 클래스 이름

        Returns:
            Optional[InferenceStrategy]: 전략 인스턴스 또는 None
        """
        return self._pipeline.get(name)

    @property
    def strategy_names(self) -> List[str]:
        """로드된 전략 이름 목록"""
        return list(self._pipeline.keys())
