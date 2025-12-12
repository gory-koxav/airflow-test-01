# -*- coding: utf-8 -*-
"""
AI Inference Service

YOLO Object Detection, SAM Segmentation, Classification 파이프라인을 실행합니다.
legacy_reference/cctv_event_detector/inference/의 로직을 Airflow 환경에 맞게 래핑합니다.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import numpy as np

from .image_provider.base import CapturedImage

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """
    AI 추론 결과 데이터 모델

    Attributes:
        image_id: 이미지 식별자
        camera_name: 카메라 이름
        detections: YOLO Object Detection 결과 리스트
        boundary_masks: SAM Segmentation 마스크 리스트
        assembly_classifications: Assembly Stage 분류 결과 리스트
        pinjig_masks: Pinjig 관련 마스크 리스트
        pinjig_classifications: Pinjig 분류 결과 리스트
        metadata: 추가 메타데이터
    """
    image_id: str
    camera_name: str
    detections: List[Dict[str, Any]] = field(default_factory=list)
    boundary_masks: List[Dict[str, Any]] = field(default_factory=list)
    assembly_classifications: List[Dict[str, Any]] = field(default_factory=list)
    pinjig_masks: List[np.ndarray] = field(default_factory=list)
    pinjig_classifications: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceService:
    """
    AI 추론 서비스

    YOLO Object Detection, SAM Segmentation, Classification 파이프라인을 실행합니다.
    """

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        sam_model_path: Optional[str] = None,
        sam_yolo_cls_model_path: Optional[str] = None,
        assembly_cls_model_path: Optional[str] = None,
        device: str = "cuda",
        yolo_confidence_threshold: float = 0.5,
        sam_confidence_threshold: float = 0.5,
    ):
        """
        Args:
            yolo_model_path: YOLO Object Detection 모델 경로
            sam_model_path: SAM Segmentation 모델 경로
            sam_yolo_cls_model_path: SAM 분류용 YOLO 모델 경로
            assembly_cls_model_path: Assembly Classification 모델 경로
            device: 추론 디바이스 ("cuda" 또는 "cpu")
            yolo_confidence_threshold: YOLO confidence threshold
            sam_confidence_threshold: SAM confidence threshold
        """
        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.sam_yolo_cls_model_path = sam_yolo_cls_model_path
        self.assembly_cls_model_path = assembly_cls_model_path
        self.device = device
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.sam_confidence_threshold = sam_confidence_threshold

        self._models_loaded = False
        self._yolo_model = None
        self._sam_model = None
        self._sam_cls_model = None
        self._assembly_cls_model = None

    def _load_models(self) -> None:
        """모델 로드 (Lazy Loading)"""
        if self._models_loaded:
            return

        logger.info("Loading AI models...")

        try:
            # YOLO Object Detection 모델 로드
            if self.yolo_model_path:
                from ultralytics import YOLO
                self._yolo_model = YOLO(self.yolo_model_path)
                logger.info(f"YOLO model loaded: {self.yolo_model_path}")

            # SAM 모델 로드
            if self.sam_model_path:
                # TODO: SAM 모델 로드 구현
                # from segment_anything import sam_model_registry
                logger.info(f"SAM model path configured: {self.sam_model_path}")

            # SAM Classification 모델 로드
            if self.sam_yolo_cls_model_path:
                from ultralytics import YOLO
                self._sam_cls_model = YOLO(self.sam_yolo_cls_model_path)
                logger.info(f"SAM CLS model loaded: {self.sam_yolo_cls_model_path}")

            # Assembly Classification 모델 로드
            if self.assembly_cls_model_path:
                from ultralytics import YOLO
                self._assembly_cls_model = YOLO(self.assembly_cls_model_path)
                logger.info(f"Assembly CLS model loaded: {self.assembly_cls_model_path}")

            self._models_loaded = True
            logger.info("All AI models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            raise

    def run_inference(
        self,
        captured_images: List[CapturedImage],
    ) -> Dict[str, InferenceResult]:
        """
        캡처된 이미지에 대해 AI 추론을 실행합니다.

        Args:
            captured_images: 캡처된 이미지 리스트

        Returns:
            Dict[str, InferenceResult]: image_id를 키로 하는 추론 결과 딕셔너리
        """
        self._load_models()

        results: Dict[str, InferenceResult] = {}

        for captured_image in captured_images:
            try:
                result = self._process_single_image(captured_image)
                results[captured_image.image_id] = result
            except Exception as e:
                logger.error(
                    f"Inference failed for {captured_image.image_id}: {e}"
                )
                # 실패한 이미지에 대해 빈 결과 생성
                results[captured_image.image_id] = InferenceResult(
                    image_id=captured_image.image_id,
                    camera_name=captured_image.camera_name,
                    metadata={"error": str(e)},
                )

        logger.info(f"Inference completed for {len(results)} images")
        return results

    def _process_single_image(
        self,
        captured_image: CapturedImage,
    ) -> InferenceResult:
        """단일 이미지에 대해 추론 실행"""
        image_data = captured_image.image_data

        # 1. YOLO Object Detection
        detections = self._run_yolo_detection(image_data)

        # 2. SAM Segmentation (boundary detection)
        boundary_masks = self._run_sam_segmentation(image_data, detections)

        # 3. Assembly Classification
        assembly_classifications = self._run_assembly_classification(
            image_data, detections
        )

        # 4. Pinjig Detection & Classification
        pinjig_masks, pinjig_classifications = self._run_pinjig_detection(
            image_data, detections
        )

        return InferenceResult(
            image_id=captured_image.image_id,
            camera_name=captured_image.camera_name,
            detections=detections,
            boundary_masks=boundary_masks,
            assembly_classifications=assembly_classifications,
            pinjig_masks=pinjig_masks,
            pinjig_classifications=pinjig_classifications,
            metadata={
                "captured_at": captured_image.captured_at.isoformat(),
                "bay_id": captured_image.bay_id,
            },
        )

    def _run_yolo_detection(
        self,
        image_data: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """YOLO Object Detection 실행"""
        if self._yolo_model is None:
            return []

        results = self._yolo_model(
            image_data,
            conf=self.yolo_confidence_threshold,
            device=self.device,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                detection = {
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "confidence": float(boxes.conf[i].cpu()),
                    "class_id": int(boxes.cls[i].cpu()),
                    "class_name": result.names[int(boxes.cls[i].cpu())],
                }
                detections.append(detection)

        return detections

    def _run_sam_segmentation(
        self,
        image_data: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """SAM Segmentation 실행"""
        # TODO: SAM 모델을 사용한 segmentation 구현
        # 현재는 placeholder로 빈 리스트 반환
        return []

    def _run_assembly_classification(
        self,
        image_data: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assembly Stage Classification 실행"""
        if self._assembly_cls_model is None:
            return []

        classifications = []

        # panel, block, HP_block 클래스만 분류
        target_classes = ["panel", "block", "HP_block"]

        for detection in detections:
            if detection["class_name"] not in target_classes:
                continue

            try:
                # detection bbox로 crop
                bbox = detection["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cropped = image_data[y1:y2, x1:x2]

                if cropped.size == 0:
                    continue

                # classification 실행
                result = self._assembly_cls_model(cropped, device=self.device)

                if result and len(result) > 0:
                    probs = result[0].probs
                    if probs is not None:
                        classification = {
                            "detection_bbox": bbox,
                            "class_id": int(probs.top1),
                            "class_name": result[0].names[probs.top1],
                            "confidence": float(probs.top1conf.cpu()),
                        }
                        classifications.append(classification)

            except Exception as e:
                logger.warning(f"Assembly classification failed: {e}")

        return classifications

    def _run_pinjig_detection(
        self,
        image_data: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> tuple:
        """Pinjig Detection & Classification 실행"""
        # TODO: Pinjig 관련 처리 구현
        return [], []

    def is_ready(self) -> bool:
        """모델이 로드되었는지 확인"""
        return self._models_loaded
