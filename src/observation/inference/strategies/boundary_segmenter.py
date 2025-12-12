# -*- coding: utf-8 -*-
"""
SAM Object Boundary Segmenter Strategy

SAM (Segment Anything Model)을 사용하여 탐지된 객체의 경계를 분할하는 전략입니다.
Legacy visionflow-legacy-observation의 object_boundary_segmenter.py를 마이그레이션했습니다.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

from src.observation.models import CapturedImage
from .base import InferenceStrategy

logger = logging.getLogger(__name__)


class SAMObjectBoundarySegmenter(InferenceStrategy):
    """
    SAM 객체 경계 분할 전략

    YOLO로 탐지된 객체(block, panel, HP_block)에 대해 SAM을 사용하여
    정밀한 경계 마스크를 생성합니다.

    Attributes:
        predictor: SAM Predictor 인스턴스
        device: 추론 디바이스
        target_classes: 경계 분할 대상 클래스 집합
        output_dir: 시각화 결과 저장 경로
    """

    def __init__(
        self,
        model_path: str = None,
        model_type: str = None,
        target_classes: List[str] = None,
        output_dir: str = None,
        device: str = None,
    ):
        """
        Args:
            model_path: SAM 모델 체크포인트 경로
            model_type: SAM 모델 타입 (예: "vit_h", "vit_l", "vit_b")
            target_classes: 경계 분할 대상 클래스 리스트
            output_dir: 시각화 결과 저장 경로
            device: 추론 디바이스 (None이면 자동 감지)
        """
        logger.info("Initializing SAM Object Boundary Segmenter...")

        # 설정 로드
        from config.settings import get_setting
        setting = get_setting()

        if model_path is None:
            model_path = str(setting.path.sam_ckpt)
        if model_type is None:
            model_type = setting.model.sam.type
        if target_classes is None:
            target_classes = setting.target.classes.boundary
        if output_dir is None:
            output_dir = str(setting.path.boundary_output)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # SAM 모델 로드
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        logger.info(f"SAM Predictor ('{model_type}') loaded successfully.")

        # 대상 클래스 설정
        self.target_classes: Set[str] = set(target_classes) if target_classes else set()
        if self.target_classes:
            logger.info(f"Target classes for boundary finding: {self.target_classes}")
        else:
            logger.info("Target classes not specified, processing all classes.")

        # 출력 디렉토리 설정
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Boundary visualization output dir: {self.output_dir}")

    def run(
        self,
        captured_images: List[CapturedImage],
        inference_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        SAM 경계 분할 실행

        Args:
            captured_images: 캡처된 이미지 리스트
            inference_results: YOLO 탐지 결과가 포함된 추론 결과

        Returns:
            Dict[str, Any]: 경계 마스크가 추가된 추론 결과 딕셔너리
        """
        logger.info("Running SAM Object Boundary Segmenter...")

        for capture in captured_images:
            image_id = capture.image_id
            original_image = capture.image_data

            detections = inference_results.get(image_id, {}).get("detections", [])
            if not detections:
                continue

            # BGR -> RGB 변환 후 SAM에 설정
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(rgb_image)

            boundary_masks = []
            vis_image = original_image.copy()

            # 시각화를 위해 모든 YOLO 탐지 결과를 먼저 그림
            for detection in detections:
                self._draw_yolo_bbox_and_label(vis_image, detection)

            for detection in detections:
                # target_classes가 비어있거나, class가 target에 포함되면 처리
                should_process = (
                    not self.target_classes
                    or detection['class_name'] in self.target_classes
                )

                if should_process:
                    mask_result = self._process_single_detection(
                        detection, vis_image
                    )
                    if mask_result is not None:
                        boundary_masks.append(mask_result)

            # 결과 저장
            if boundary_masks:
                output_dir_path = self.output_dir / image_id
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_path = output_dir_path / f"{image_id}_boundaries.jpg"
                cv2.imwrite(str(output_path), vis_image)
                logger.info(f"Saved boundary visualization to {output_path}")

                if "boundary_masks" not in inference_results[image_id]:
                    inference_results[image_id]['boundary_masks'] = []
                inference_results[image_id]['boundary_masks'].extend(boundary_masks)

        return inference_results

    def _process_single_detection(
        self,
        detection: Dict[str, Any],
        vis_image: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        단일 탐지 객체에 대해 SAM 분할 수행

        Args:
            detection: YOLO 탐지 결과
            vis_image: 시각화용 이미지 (in-place 수정)

        Returns:
            Optional[Dict[str, Any]]: 분할 결과 또는 None
        """
        # detector는 [x_min, y_min, width, height] format을 사용
        x_min, y_min, w, h = detection['bbox_xywh']

        # SAM은 [x1, y1, x2, y2] 즉 (좌상단, 우하단) 포맷을 요구
        input_box = np.array([x_min, y_min, x_min + w, y_min + h])

        # SAM 추론
        masks, scores, _ = self.predictor.predict(
            box=input_box,
            multimask_output=False
        )
        mask = masks[0]

        # 마스크 후처리 (morphological operations)
        processed_mask = self._postprocess_mask(mask, detection)

        if processed_mask is not None:
            # 시각화에 마스크 오버레이
            self._draw_sam_mask_on_image(vis_image, processed_mask)

            return {
                "detected_bbox_idx": detection['detected_bbox_idx'],
                "class_name": detection['class_name'],
                "confidence": float(scores[0]),
                "segmentation_mask": processed_mask,
            }

        return None

    def _postprocess_mask(
        self,
        mask: np.ndarray,
        detection: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """
        마스크 후처리 (노이즈 제거, 가장 큰 폐곡선 선택)

        Args:
            mask: SAM 출력 마스크
            detection: 탐지 정보 (로깅용)

        Returns:
            Optional[np.ndarray]: 후처리된 마스크 또는 None
        """
        # Morphological operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seg = mask.astype(np.uint8)

        seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel_open)    # 노이즈 제거
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel_close)  # 구멍 채우기

        # 가장 큰 폐곡선만 유지
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            closed_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 0 and (cv2.isContourConvex(contour) or len(contour) >= 3):
                    if (np.array_equal(contour[0], contour[-1])
                            or cv2.arcLength(contour, True) > 0):
                        closed_contours.append(contour)

            if closed_contours:
                largest_contour = max(closed_contours, key=cv2.contourArea)
                seg = np.zeros_like(seg)
                cv2.fillPoly(seg, [largest_contour], 255)
                return seg.astype(bool)
            else:
                logger.warning(
                    f"No closed contours found for detection "
                    f"{detection['detected_bbox_idx']}. Using original SAM mask."
                )
                return mask
        else:
            logger.warning(
                f"No valid contours found for detection "
                f"{detection['detected_bbox_idx']}. Using original SAM mask."
            )
            return mask

    def _draw_yolo_bbox_and_label(
        self,
        image: np.ndarray,
        detection: Dict[str, Any],
    ) -> None:
        """이미지에 YOLO 바운딩 박스와 클래스 레이블을 그립니다."""
        x_min, y_min, w, h = map(int, detection['bbox_xywh'])
        class_name = detection['class_name']

        x1, y1 = x_min, y_min
        x2, y2 = x_min + w, y_min + h

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (label_w, label_h), _ = cv2.getTextSize(
            f"{class_name}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(image, (x1, y1 - 20), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(
            image, f"{class_name}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    def _draw_sam_mask_on_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """주어진 이미지에 SAM 마스크를 랜덤 색상으로 오버레이합니다."""
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        overlay = image.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
