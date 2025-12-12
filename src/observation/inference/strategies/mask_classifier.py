# -*- coding: utf-8 -*-
"""
Mask Classifier Strategy

입력된 마스크를 기반으로 객체를 크롭하고 YOLO 분류 모델로 추론한 뒤
결과를 시각화하여 저장하는 전략입니다.
Legacy visionflow-legacy-observation의 mask_classifier.py를 마이그레이션했습니다.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
from ultralytics import YOLO

from src.observation.models import CapturedImage
from .base import InferenceStrategy

logger = logging.getLogger(__name__)


class MaskClassifier(InferenceStrategy):
    """
    마스크 분류 전략

    AutomaticSegmenter에서 생성된 마스크를 기반으로 객체를 크롭하고,
    YOLO 분류 모델로 추론한 뒤 결과를 시각화하여 저장합니다.
    이 클래스는 '분류 및 시각화'의 책임만 가집니다.

    Attributes:
        cls_model: YOLO 분류 모델 인스턴스
        output_dir: 분류 결과 시각화 저장 경로
        device: 추론 디바이스
    """

    def __init__(
        self,
        model_path: str = None,
        output_dir: str = None,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: YOLO 분류 모델 체크포인트 경로
            output_dir: 분류 결과 시각화 저장 경로
            device: 추론 디바이스
        """
        logger.info("Initializing Mask Classifier...")

        # 설정 로드
        from config.settings import get_setting
        setting = get_setting()

        if model_path is None:
            model_path = str(setting.path.yolo_cls_ckpt)
        if output_dir is None:
            output_dir = str(setting.path.sam_output)

        self.cls_model = YOLO(model_path)
        self.device = device
        logger.info(f"YOLO classification model loaded successfully: {model_path}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Classification output dir: {self.output_dir}")

    def run(
        self,
        captured_images: List[CapturedImage],
        inference_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        마스크 분류 실행

        Args:
            captured_images: 캡처된 이미지 리스트
            inference_results: 핀지그 마스크가 포함된 추론 결과

        Returns:
            Dict[str, Any]: 분류 결과가 추가된 추론 결과 딕셔너리
        """
        logger.info("Running Mask Classifier...")

        for capture in captured_images:
            image_id = capture.image_id
            original_image = capture.image_data

            masks = inference_results.get(image_id, {}).get("pinjig_masks", [])

            if masks:
                classification_results = self._perform_classification(
                    original_image, masks, image_id
                )

                if classification_results:
                    inference_results[image_id]['pinjig_classifications'] = (
                        classification_results
                    )
                    self._visualize_results(
                        original_image, masks, classification_results, image_id
                    )

        return inference_results

    def _perform_classification(
        self,
        original_image: np.ndarray,
        masks: List[Dict[str, Any]],
        image_id: str,
    ) -> List[Dict[str, Any]]:
        """
        마스크별 분류 수행

        Args:
            original_image: 원본 이미지
            masks: 분할된 마스크 리스트
            image_id: 이미지 ID

        Returns:
            List[Dict[str, Any]]: 분류 결과 리스트
        """
        timed_output_dir = self.output_dir / image_id
        timed_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Classifying {len(masks)} masks for '{image_id}'...")

        classifications = []
        for i, mask_ann in enumerate(masks):
            bbox = mask_ann.get('bbox')
            if not bbox:
                continue

            try:
                x, y, w, h = map(int, bbox)
            except (ValueError, TypeError):
                logger.warning(f"Skipping mask with invalid bbox data: {bbox}")
                continue

            if w <= 0 or h <= 0:
                logger.warning(f"Skipping mask with invalid bbox dimensions: w={w}, h={h}")
                continue

            # 마스크 영역 크롭
            rect_crop = original_image[y:y+h, x:x+w]
            segmentation = mask_ann.get('segmentation')

            if segmentation is None:
                continue

            cropped_mask = segmentation[y:y+h, x:x+w]
            black_background = np.zeros_like(rect_crop)
            black_background[cropped_mask] = rect_crop[cropped_mask]

            if black_background.size == 0:
                continue

            # 분류 추론
            resized_image = cv2.resize(black_background, (512, 512))
            results = self.cls_model(resized_image, device=self.device, verbose=False)
            r = results[0]

            top1_class_name = self.cls_model.names[r.probs.top1]
            top1_conf = r.probs.top1conf.item()

            color = np.random.uniform(0, 255, 3).tolist()

            classifications.append({
                "mask_index": i,
                "top1_class": top1_class_name,
                "confidence": top1_conf,
                "top5_probs": r.probs.top5conf.tolist(),
                "color": color,
            })

            # 개별 분류 결과 저장
            annotated_image = self._annotate_image(resized_image, r)
            output_filename = f"{top1_class_name}_mask_{i:03d}.png"
            cv2.imwrite(str(timed_output_dir / output_filename), annotated_image)

        return classifications

    def _visualize_results(
        self,
        original_image: np.ndarray,
        masks: List[Dict[str, Any]],
        classifications: List[Dict[str, Any]],
        image_id: str,
    ) -> None:
        """
        분류 결과 시각화

        Args:
            original_image: 원본 이미지
            masks: 분할된 마스크 리스트
            classifications: 분류 결과 리스트
            image_id: 이미지 ID
        """
        timed_output_dir = self.output_dir / image_id
        overlay_image = original_image.copy()

        img_h, img_w = overlay_image.shape[:2]

        # 1단계: 모든 마스크 그리기
        for classification in classifications:
            mask_index = classification.get('mask_index')
            if mask_index is None or mask_index >= len(masks):
                continue

            mask = masks[mask_index].get('segmentation')
            if mask is None:
                continue

            color = np.array(classification['color'])
            overlay_image[mask] = overlay_image[mask] * 0.4 + color * 0.6

            # bbox를 마스크와 동일한 색상으로 그리기
            bbox = masks[mask_index].get('bbox')
            if bbox:
                x, y, w, h = map(int, bbox)
                color_int = tuple(map(int, color))
                cv2.rectangle(overlay_image, (x, y), (x + w, y + h), color_int, 2)

        # 2단계: 연결선 및 텍스트 그리기
        for classification in classifications:
            mask_index = classification.get('mask_index')
            if mask_index is None or mask_index >= len(masks):
                continue

            mask = masks[mask_index].get('segmentation')
            bbox = masks[mask_index].get('bbox')
            if mask is None or bbox is None:
                continue

            x, y, w, h = map(int, bbox)
            color = classification['color']

            # 마스크 중심점 계산
            try:
                M = cv2.moments(mask.astype(np.uint8))
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = x + w // 2, y + h // 2
            except ZeroDivisionError:
                cX, cY = x + w // 2, y + h // 2

            # 텍스트 정보
            text = f"{classification['top1_class']} ({classification['confidence']:.2f})"
            font_scale = 0.6
            font_thickness = 1
            (w_text, h_text), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # 텍스트 위치 계산 (이미지 경계 보정)
            ideal_text_pos_x = x
            ideal_text_pos_y = y - 10 if y > (h_text + 20) else y + h + h_text + 10

            rect_x1 = ideal_text_pos_x - 5
            rect_y1 = ideal_text_pos_y - h_text - 5
            rect_x2 = ideal_text_pos_x + w_text + 5
            rect_y2 = ideal_text_pos_y + 5

            # 경계 보정
            dx = 0
            if rect_x1 < 0:
                dx = -rect_x1
            if rect_x2 > img_w:
                dx = img_w - rect_x2

            dy = 0
            if rect_y1 < 0:
                dy = -rect_y1
            if rect_y2 > img_h:
                dy = img_h - rect_y2

            final_text_pos = (ideal_text_pos_x + dx, ideal_text_pos_y + dy)
            final_rect_pos1 = (rect_x1 + dx, rect_y1 + dy)
            final_rect_pos2 = (rect_x2 + dx, rect_y2 + dy)

            # 연결선 그리기
            line_start_point = (
                final_text_pos[0] + w_text // 2,
                final_rect_pos2[1] if final_text_pos[1] > y else final_rect_pos1[1]
            )
            cv2.line(overlay_image, line_start_point, (cX, cY), (255, 255, 255), 5)
            cv2.line(overlay_image, line_start_point, (cX, cY), tuple(map(int, color)), 3)

            # 배경 사각형 그리기
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
            cv2.rectangle(
                overlay_image, final_rect_pos1, final_rect_pos2,
                tuple(map(int, color)), -1
            )

            # 텍스트 그리기
            cv2.putText(
                overlay_image, text, final_text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                font_thickness, cv2.LINE_AA
            )

        cv2.imwrite(str(timed_output_dir / "_OVERLAY_RESULT.jpg"), overlay_image)
        logger.info(f"Saved overlay result: {timed_output_dir / '_OVERLAY_RESULT.jpg'}")

    def _annotate_image(self, image: np.ndarray, result) -> np.ndarray:
        """분류 결과를 이미지에 어노테이션"""
        annotated_image = image.copy()
        pos_y = 40
        for j, idx in enumerate(result.probs.top5):
            text = f"{j+1}. {self.cls_model.names[int(idx)]}: {result.probs.top5conf[j]:.2f}"
            cv2.putText(
                annotated_image, text, (20, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                annotated_image, text, (20, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
            )
            pos_y += 40
        return annotated_image
