# -*- coding: utf-8 -*-
"""
YOLO Object Detector Strategy

YOLO 모델을 사용한 객체 탐지 전략입니다.
Legacy visionflow-legacy-observation의 object_detector.py를 마이그레이션했습니다.
"""

import logging
from typing import Dict, Any, List
from collections import Counter

import numpy as np
from ultralytics import YOLO

from src.observation.models import CapturedImage
from .base import InferenceStrategy

logger = logging.getLogger(__name__)


class YOLOObjectDetector(InferenceStrategy):
    """
    YOLO 객체 탐지 전략

    YOLO 모델을 사용하여 이미지에서 객체를 탐지합니다.
    탐지된 객체의 바운딩 박스, 클래스, 신뢰도를 반환합니다.

    Attributes:
        model: YOLO 모델 인스턴스
        confidence_threshold: 탐지 신뢰도 임계값
        device: 추론 디바이스 ("cuda" 또는 "cpu")
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: YOLO 모델 체크포인트 경로
                None이면 config.settings에서 기본 경로 로드
            confidence_threshold: 탐지 신뢰도 임계값
            device: 추론 디바이스
        """
        if model_path is None:
            from config.settings import get_setting
            model_path = str(get_setting().path.yolo_od_ckpt)

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        logger.info(f"YOLO Object Detection model loaded from: {model_path}")

    def run(
        self,
        captured_images: List[CapturedImage],
        inference_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        YOLO 객체 탐지 실행

        Args:
            captured_images: 캡처된 이미지 리스트
            inference_results: 이전 전략들의 추론 결과

        Returns:
            Dict[str, Any]: 탐지 결과가 추가된 추론 결과 딕셔너리
        """
        logger.info("Running YOLO Object Detector...")

        image_data_list = [img.image_data for img in captured_images]
        if not image_data_list:
            return inference_results

        # YOLO 추론 실행
        yolo_results = self.model.predict(
            image_data_list,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        # 결과 정리
        for i, capture in enumerate(captured_images):
            image_id = capture.image_id
            if image_id not in inference_results:
                inference_results[image_id] = {}

            result = yolo_results[i]
            detected_objects = []

            for idx, box in enumerate(result.boxes):
                # YOLO는 중심점 기준 좌표를 반환하므로 좌상단 기준으로 좌표 변환
                center_x, center_y, width, height = box.xywh[0].tolist()

                x_min = center_x - width / 2
                y_min = center_y - height / 2

                # 반올림하여 정수로 변환
                bbox_xywh = [round(x_min), round(y_min), round(width), round(height)]

                detected_objects.append({
                    "detected_bbox_idx": idx,
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox_xywh": bbox_xywh,
                })

            inference_results[image_id]["detections"] = detected_objects

            # 로깅
            total_objects = len(detected_objects)
            if total_objects > 0:
                class_counts = Counter(obj['class_name'] for obj in detected_objects)
                class_summary = ", ".join(
                    [f"{name} ({count})" for name, count in class_counts.items()]
                )
                logger.info(f"Found {total_objects} objects in '{image_id}': {class_summary}")
            else:
                logger.info(f"Found 0 objects in '{image_id}'")

        return inference_results
