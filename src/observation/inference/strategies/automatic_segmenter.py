# -*- coding: utf-8 -*-
"""
Automatic Segmenter Strategy

SAM의 AutomaticMaskGenerator를 사용하여 이미지에서 핀지그 등의 객체 마스크를
자동으로 생성하는 전략입니다.
Legacy visionflow-legacy-observation의 automatic_segmenter.py를 마이그레이션했습니다.
"""

import logging
from typing import Dict, Any, List

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from src.observation.models import CapturedImage
from .base import InferenceStrategy

logger = logging.getLogger(__name__)


class AutomaticSegmenter(InferenceStrategy):
    """
    SAM 자동 분할 전략

    SAM의 AutomaticMaskGenerator를 사용하여 이미지에서 객체 마스크들을 자동으로 생성합니다.
    주로 핀지그(pinjig) 등의 작은 객체를 탐지하는 데 사용됩니다.
    이 클래스는 '분할'의 책임만 가지며, 분류는 MaskClassifier에서 수행합니다.

    Attributes:
        mask_generator: SAM AutomaticMaskGenerator 인스턴스
        device: 추론 디바이스
        cutoff_pct_top: 이미지 상단 회색 패딩 비율
        cutoff_pct_bot: 이미지 하단 회색 패딩 비율
    """

    def __init__(
        self,
        model_path: str = None,
        model_type: str = None,
        cutoff_pct_top: float = None,
        cutoff_pct_bot: float = None,
        device: str = None,
    ):
        """
        Args:
            model_path: SAM 모델 체크포인트 경로
            model_type: SAM 모델 타입 (예: "vit_h", "vit_l", "vit_b")
            cutoff_pct_top: 이미지 상단 회색 패딩 비율 (0-100)
            cutoff_pct_bot: 이미지 하단 회색 패딩 비율 (0-100)
            device: 추론 디바이스 (None이면 자동 감지)
        """
        logger.info("Initializing Automatic Segmenter...")

        # 설정 로드
        from config.settings import get_setting
        setting = get_setting()

        if model_path is None:
            model_path = str(setting.path.sam_ckpt)
        if model_type is None:
            model_type = setting.model.sam.type
        if cutoff_pct_top is None:
            cutoff_pct_top = setting.model.sam.cutoff_pct_top
        if cutoff_pct_bot is None:
            cutoff_pct_bot = setting.model.sam.cutoff_pct_bot

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cutoff_pct_top = cutoff_pct_top
        self.cutoff_pct_bot = cutoff_pct_bot

        # SAM 모델 로드
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=sam)
        logger.info(f"Automatic SAM model ('{model_type}') loaded successfully.")

    def run(
        self,
        captured_images: List[CapturedImage],
        inference_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        자동 분할 실행

        Args:
            captured_images: 캡처된 이미지 리스트
            inference_results: 이전 전략들의 추론 결과

        Returns:
            Dict[str, Any]: 핀지그 마스크가 추가된 추론 결과 딕셔너리
        """
        logger.info("Running Automatic Segmenter...")

        for capture in captured_images:
            image_id = capture.image_id
            image_data = capture.image_data
            h, w, _ = image_data.shape

            # 이미지 전처리 (가우시안 블러)
            processed_image = cv2.GaussianBlur(image_data, (31, 31), 0)

            # 마스크 생성
            logger.info(f"Generating masks for {image_id}...")
            raw_masks = self.mask_generator.generate(processed_image)

            # 특정 영역 무시를 위한 마스크 생성
            ignore_mask = np.zeros((h, w), dtype=bool)

            # 마스크 필터링
            filtered_masks = self._filter_masks(raw_masks, ignore_mask)
            logger.info(f"Found {len(filtered_masks)} final masks for {image_id}.")

            # 결과 저장
            if image_id not in inference_results:
                inference_results[image_id] = {}
            inference_results[image_id]['pinjig_masks'] = filtered_masks

        return inference_results

    def _apply_top_bottom_gray_padding(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        이미지의 상단과 하단을 지정된 비율만큼 회색으로 채웁니다.

        Args:
            image_rgb: 원본 RGB 이미지 배열

        Returns:
            np.ndarray: 상단과 하단이 회색으로 채워진 수정된 이미지 배열
        """
        padded_image = image_rgb.copy()
        height, _ = padded_image.shape[:2]

        top_end_y = int(height * (self.cutoff_pct_top / 100.0))
        bottom_start_y = int(height * (self.cutoff_pct_bot / 100.0))

        gray_color = [114, 114, 114]
        padded_image[0:top_end_y, :] = gray_color
        padded_image[bottom_start_y:height, :] = gray_color

        return padded_image

    def _filter_masks(
        self,
        masks: List[Dict[str, Any]],
        ignore_mask: np.ndarray,
        nesting_threshold: float = 0.9,
        ignore_iou_threshold: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        마스크 필터링 (노이즈 제거, 중첩 마스크 제거)

        Args:
            masks: SAM이 생성한 원본 마스크 리스트
            ignore_mask: 무시할 영역 마스크
            nesting_threshold: 중첩 마스크 제거 임계값
            ignore_iou_threshold: 무시 영역과의 IoU 임계값

        Returns:
            List[Dict[str, Any]]: 필터링된 마스크 리스트
        """
        if not masks:
            return []

        # 무시 영역과 겹치는 마스크 제거
        masks_in_roi = [
            m for m in masks
            if np.logical_and(m['segmentation'], ignore_mask).sum() / m['area']
            < ignore_iou_threshold
        ]
        if not masks_in_roi:
            return []

        # 면적 기준 내림차순 정렬
        sorted_masks = sorted(masks_in_roi, key=(lambda x: x['area']), reverse=True)

        # 마스크 후처리 (morphological operations)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for mask in sorted_masks:
            seg = mask['segmentation'].astype(np.uint8)
            seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel_open)
            seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel_close)

            # 가장 큰 폐곡선만 유지
            contours, _ = cv2.findContours(
                seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            largest_contour = None
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

                    # 바운딩 박스 재계산
                    x, y, w, h = cv2.boundingRect(seg)
                    mask['bbox'] = [x, y, w, h]
                else:
                    seg = np.zeros_like(seg)

            mask['segmentation'] = seg.astype(bool)
            mask['area'] = seg.sum()
            if largest_contour is not None:
                mask['coords'] = largest_contour.squeeze()

        # 중첩 마스크 제거
        final_masks_indices = list(range(len(sorted_masks)))

        for i in range(len(sorted_masks)):
            if i not in final_masks_indices:
                continue
            mask_i_seg = sorted_masks[i]['segmentation']

            for j in range(i + 1, len(sorted_masks)):
                if j not in final_masks_indices:
                    continue
                mask_j_seg = sorted_masks[j]['segmentation']
                intersection = np.logical_and(mask_i_seg, mask_j_seg).sum()

                if sorted_masks[j]['area'] > 0:
                    if intersection / sorted_masks[j]['area'] > nesting_threshold:
                        final_masks_indices.remove(j)

        # 최소 크기 필터링 (10x10 이상)
        return [
            sorted_masks[i] for i in final_masks_indices
            if (sorted_masks[i].get('bbox') and
                sorted_masks[i]['bbox'][2] >= 10 and
                sorted_masks[i]['bbox'][3] >= 10)
        ]
