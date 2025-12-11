# -*- coding: utf-8 -*-
"""
Fusion Service

다중 카메라 데이터를 융합하여 블록 정보를 생성하는 메인 서비스입니다.
Observation 결과 (Pickle)를 읽어서 처리하고 결과를 저장합니다.
"""

import pickle
import gzip
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
import numpy as np
import cv2

from .models import FrameData, ProjectedData, MergedBlockInfo
from .projector import Projector
from .overlap_analyzer import OverlapAnalyzer

logger = logging.getLogger(__name__)

# 무효한 투표 값 목록
INVALID_VOTING_VALUES = {"", "none", "null", "n/a", "unknown", "-", "nan"}
MIN_VALID_VALUE_LENGTH = 2


class FusionService:
    """
    다중 카메라 데이터를 융합하여 블록의 공간 정보를 분석하는 서비스

    Redis pub-sub 대신 Pickle 파일 기반으로 데이터를 교환합니다.
    """

    def __init__(
        self,
        bay_id: str,
        camera_intrinsics: Dict[str, float],
        camera_extrinsics: Dict[str, Dict[str, Any]],
        boundary_target_classes: List[str],
        base_path: str = "/opt/airflow/data",
        iou_threshold: float = 0.3,
        pixels_per_meter: int = 10,
    ):
        """
        Args:
            bay_id: Bay 식별자
            camera_intrinsics: 카메라 내부 파라미터
            camera_extrinsics: 카메라별 외부 파라미터
            boundary_target_classes: Boundary로 처리할 객체 클래스 목록
            base_path: 데이터 저장 기본 경로
            iou_threshold: IOU 임계값
            pixels_per_meter: 미터당 픽셀 수
        """
        self.bay_id = bay_id
        self.base_path = Path(base_path)
        self.boundary_target_classes = set(boundary_target_classes)
        self.pixels_per_meter = pixels_per_meter

        # Projector 및 OverlapAnalyzer 초기화
        self.projector = Projector(
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            pixels_per_meter=pixels_per_meter,
        )
        self.overlap_analyzer = OverlapAnalyzer(iou_threshold=iou_threshold)

        logger.info(f"[{bay_id}] FusionService 초기화 완료")

    def load_observation_result(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Observation 결과를 Pickle 파일에서 로드

        Args:
            batch_id: 배치 식별자

        Returns:
            로드된 데이터 또는 None
        """
        result_path = (
            self.base_path / "observation" / self.bay_id / batch_id / "result.pkl"
        )

        try:
            if not result_path.exists():
                logger.error(f"[{self.bay_id}] 결과 파일 없음: {result_path}")
                return None

            # 압축 여부 자동 감지
            try:
                with gzip.open(result_path, "rb") as f:
                    data = pickle.load(f)
            except gzip.BadGzipFile:
                with open(result_path, "rb") as f:
                    data = pickle.load(f)

            logger.info(f"[{self.bay_id}] Observation 결과 로드 완료: {batch_id}")
            return data

        except Exception as e:
            logger.error(f"[{self.bay_id}] 결과 로드 실패: {e}")
            return None

    def _convert_to_frame_data(
        self, observation_data: Dict[str, Any]
    ) -> List[FrameData]:
        """
        Observation 데이터를 FrameData 리스트로 변환

        Args:
            observation_data: Observation 결과 데이터

        Returns:
            FrameData 리스트
        """
        frame_data_list = []
        images = observation_data.get("images", {})
        inference_results = observation_data.get("inference_results", {})

        for image_id, img_info in images.items():
            result = inference_results.get(image_id, {})

            # 이미지 shape 추출
            image_data = img_info.get("image_data")
            if image_data is not None:
                image_shape = image_data.shape[:2]
            else:
                image_shape = (1080, 1920)  # 기본값

            # captured_at 파싱
            captured_at_str = img_info.get("captured_at", "")
            try:
                captured_at = datetime.fromisoformat(captured_at_str)
            except ValueError:
                captured_at = datetime.now()

            # boundary_masks 복원 (numpy 배열로)
            boundary_masks = []
            for mask_info in result.get("boundary_masks", []):
                mask_data = mask_info.get("boundary_mask")
                if isinstance(mask_data, list):
                    mask_data = np.array(mask_data, dtype=np.uint8)
                boundary_masks.append(
                    {"boundary_mask": mask_data, **{k: v for k, v in mask_info.items() if k != "boundary_mask"}}
                )

            # pinjig_masks 복원
            pinjig_masks = []
            for mask in result.get("pinjig_masks", []):
                if isinstance(mask, list):
                    mask = np.array(mask, dtype=np.uint8)
                pinjig_masks.append(mask)

            frame_data = FrameData(
                image_id=image_id,
                camera_name=img_info.get("camera_name", ""),
                image_path=f"{self.base_path}/images/{image_id}.jpg",  # 실제 경로로 조정 필요
                captured_at=captured_at,
                image_shape=image_shape,
                detections=result.get("detections", []),
                boundary_masks=boundary_masks,
                assembly_classifications=result.get("assembly_classifications", []),
                pinjig_masks=pinjig_masks,
                pinjig_classifications=result.get("pinjig_classifications", []),
            )
            frame_data_list.append(frame_data)

        return frame_data_list

    def _is_valid_value(self, value: Any) -> bool:
        """투표에 유효한 값인지 확인"""
        if value is None:
            return False
        str_value = str(value).strip().lower()
        if not str_value or str_value in INVALID_VOTING_VALUES:
            return False
        if len(str_value) < MIN_VALID_VALUE_LENGTH:
            return False
        return True

    def _vote_for_best_value(
        self, values: List[str], field_name: str
    ) -> Tuple[str, Dict[str, int]]:
        """값 목록에서 투표로 최빈값 선택"""
        if not values:
            return "", {}

        valid_values = [
            str(v).strip() for v in values if self._is_valid_value(v)
        ]
        if not valid_values:
            return "", {}

        vote_counter = Counter(valid_values)
        counts_dict = dict(vote_counter)
        most_common = vote_counter.most_common(1)

        if most_common:
            winner, count = most_common[0]
            logger.debug(
                f"{field_name} 투표 결과: '{winner}' (득표: {count}/{len(valid_values)})"
            )
            return winner, counts_dict

        return "", counts_dict

    def _initialize_block_infos(
        self, projected_results: List[ProjectedData]
    ) -> List[Dict[str, Any]]:
        """ProjectedData에서 블록 정보 초기화"""
        block_infos = []

        for proj_data in projected_results:
            if not proj_data.is_valid:
                continue

            if len(proj_data.projected_boxes_labels) != len(proj_data.projected_boxes):
                logger.warning(
                    f"[{proj_data.camera_name}] 레이블과 박스 개수 불일치"
                )
                continue

            # Boundary class에 해당하는 인덱스 찾기
            boundary_indices = [
                i
                for i, label in enumerate(proj_data.projected_boxes_labels)
                if label in self.boundary_target_classes
            ]

            if len(boundary_indices) != len(proj_data.warped_masks):
                logger.warning(
                    f"[{proj_data.camera_name}] Boundary 객체 수와 마스크 수 불일치"
                )
                continue

            for mask_idx, box_idx in enumerate(boundary_indices):
                label = proj_data.projected_boxes_labels[box_idx]
                box = proj_data.projected_boxes[box_idx]
                warped_mask = (
                    proj_data.warped_masks[mask_idx]
                    if mask_idx < len(proj_data.warped_masks)
                    else None
                )

                # Assembly stage label 찾기
                stage_label = ""
                for assembly_label_info in proj_data.projected_assembly_labels:
                    if assembly_label_info.get("detected_bbox_idx") == box_idx:
                        stage_label = assembly_label_info.get("label", "")
                        break

                # 투영 전 bbox 정보
                before_projected_bbox_xywh = []
                if box_idx < len(proj_data.before_projected_bboxes_xywh):
                    before_projected_bbox_xywh = proj_data.before_projected_bboxes_xywh[
                        box_idx
                    ]

                block_info = {
                    "vision_sensor_id": proj_data.camera_name,
                    "top1_class": label,
                    "bbox_coordinate": box,
                    "before_projected_bbox_xywh": before_projected_bbox_xywh,
                    "warped_mask": warped_mask,
                    "stage_label": stage_label,
                    "image_path": proj_data.image_path,
                    "matched_text_recognitions": [],
                    "matched_object_counts": {},
                    "matched_ships_counts": {},
                    "matched_blocks_counts": {},
                    "source_info": {
                        "camera_name": proj_data.camera_name,
                        "projected_boxes_labels_idx": box_idx,
                    },
                    "is_merged": 0,
                    "_extent": proj_data.extent,
                    "_canvas_shape": warped_mask.shape if warped_mask is not None else (0, 0),
                }
                block_infos.append(block_info)

        return block_infos

    def _create_merged_blocks(
        self,
        block_infos: List[Dict[str, Any]],
        projected_results: List[ProjectedData],
    ) -> List[Dict[str, Any]]:
        """병합된 블록 정보 생성"""
        merge_group_mapping = {}  # merge_group_id -> [block_info_indices]

        for i, block_info in enumerate(block_infos):
            camera_name = block_info["vision_sensor_id"]

            # 해당 카메라의 ProjectedData 찾기
            proj_data = None
            for pd in projected_results:
                if pd.camera_name == camera_name:
                    proj_data = pd
                    break

            if proj_data is None:
                continue

            # Boundary class 인덱스 찾기
            source_info = block_info.get("source_info", {})
            box_idx = source_info.get("projected_boxes_labels_idx")

            if box_idx is None:
                continue

            # 매핑 정보에서 그룹 ID 찾기
            boundary_indices = [
                idx
                for idx, label in enumerate(proj_data.projected_boxes_labels)
                if label in self.boundary_target_classes
            ]

            try:
                mask_idx = boundary_indices.index(box_idx)
            except ValueError:
                continue

            merge_group_id = proj_data.mask_merge_mapping.get(
                mask_idx, f"single_{camera_name}_{mask_idx}"
            )

            if merge_group_id not in merge_group_mapping:
                merge_group_mapping[merge_group_id] = []
            merge_group_mapping[merge_group_id].append(i)

        # 그룹별로 블록 정보 생성
        merged_block_infos = []
        layout_index_counter = 0

        for merge_group_id, block_indices in merge_group_mapping.items():
            blocks_to_merge = [block_infos[i] for i in block_indices]

            if len(block_indices) > 1:
                # 여러 블록 병합
                all_text_recognitions = sum(
                    [b["matched_text_recognitions"] for b in blocks_to_merge], []
                )

                ship_values = [
                    text["ship"]
                    for text in all_text_recognitions
                    if "ship" in text
                ]
                block_values = [
                    text["block"]
                    for text in all_text_recognitions
                    if "block" in text
                ]

                voted_ship, ships_counts = self._vote_for_best_value(
                    ship_values, "ship"
                )
                voted_block, blocks_counts = self._vote_for_best_value(
                    block_values, "block"
                )

                # 객체 카운트 합산
                merged_object_counts = {}
                for block in blocks_to_merge:
                    for label, count in block["matched_object_counts"].items():
                        if label not in merged_object_counts:
                            merged_object_counts[label] = 0
                        merged_object_counts[label] += count

                merged_source_infos = [b["source_info"] for b in blocks_to_merge]
                image_paths = [b["image_path"] for b in blocks_to_merge]
                original_bbox_coordinates = [b["bbox_coordinate"] for b in blocks_to_merge]
                before_projected_bboxes_xywh = [
                    b["before_projected_bbox_xywh"] for b in blocks_to_merge
                ]
                stage_labels = [
                    b.get("stage_label", "") for b in blocks_to_merge if b.get("stage_label")
                ]

                # 병합 박스 생성
                all_boxes = [b["bbox_coordinate"] for b in blocks_to_merge if b["bbox_coordinate"] is not None]
                if all_boxes:
                    all_points = np.vstack(all_boxes)
                    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
                    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
                    merged_box = np.array(
                        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                    )
                else:
                    merged_box = np.array([])

                # warped_mask 병합 (간략화)
                merge_warped_mask = None
                merge_extent = [0, 0, 0, 0]

                merged_info = {
                    "vision_sensor_id": [b["vision_sensor_id"] for b in blocks_to_merge],
                    "top1_class": [b["top1_class"] for b in blocks_to_merge],
                    "bbox_coordinate": merged_box,
                    "warped_mask": merge_warped_mask,
                    "warped_mask_shape": None,
                    "warped_mask_contour": [],
                    "stage_labels": stage_labels,
                    "image_path": image_paths[0] if image_paths else "",
                    "image_paths": image_paths,
                    "original_bbox_coordinates": original_bbox_coordinates,
                    "before_projected_bboxes_xywh": before_projected_bboxes_xywh,
                    "matched_text_recognitions": all_text_recognitions,
                    "matched_object_counts": merged_object_counts,
                    "matched_ships_counts": ships_counts,
                    "matched_blocks_counts": blocks_counts,
                    "source_infos": merged_source_infos,
                    "is_merged": 1,
                    "merged_block_idx": [i + 1 for i in block_indices],
                    "voted_ship": voted_ship,
                    "voted_block": voted_block,
                    "block_idx_in_layout": f"id_{layout_index_counter}",
                    "merge_group_id": merge_group_id,
                    "_extent": merge_extent,
                }
                merged_block_infos.append(merged_info)
            else:
                # 단일 블록
                original_info = blocks_to_merge[0].copy()
                original_info["merged_block_idx"] = []

                # 투표 실시
                text_recognitions = original_info.get("matched_text_recognitions", [])
                ship_values = [
                    text["ship"] for text in text_recognitions if "ship" in text
                ]
                block_values = [
                    text["block"] for text in text_recognitions if "block" in text
                ]

                voted_ship, ships_counts = self._vote_for_best_value(ship_values, "ship")
                voted_block, blocks_counts = self._vote_for_best_value(
                    block_values, "block"
                )

                original_info["voted_ship"] = voted_ship
                original_info["voted_block"] = voted_block
                original_info["matched_ships_counts"] = ships_counts
                original_info["matched_blocks_counts"] = blocks_counts

                # 리스트 형태로 변환
                stage_label = original_info.get("stage_label", "")
                original_info["stage_labels"] = [stage_label] if stage_label else []

                if "source_info" in original_info:
                    original_info["source_infos"] = [original_info["source_info"]]
                    del original_info["source_info"]

                original_info["image_paths"] = [original_info["image_path"]]
                original_info["original_bbox_coordinates"] = [
                    original_info["bbox_coordinate"]
                ]
                original_info["before_projected_bboxes_xywh"] = [
                    original_info["before_projected_bbox_xywh"]
                ]
                original_info["vision_sensor_id"] = [original_info["vision_sensor_id"]]
                original_info["top1_class"] = [original_info["top1_class"]]
                original_info["warped_mask_shape"] = None
                original_info["warped_mask_contour"] = []
                original_info["block_idx_in_layout"] = f"id_{layout_index_counter}"
                original_info["merge_group_id"] = merge_group_id

                merged_block_infos.append(original_info)

            layout_index_counter += 1

        return merged_block_infos

    def process_batch(
        self, batch_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        배치 데이터 처리

        Args:
            batch_id: 배치 식별자

        Returns:
            (block_infos, merged_block_infos) 튜플
        """
        logger.info(f"[{self.bay_id}] Batch '{batch_id}' 처리 시작")

        # 1. Observation 결과 로드
        observation_data = self.load_observation_result(batch_id)
        if not observation_data:
            logger.error(f"[{self.bay_id}] Observation 데이터 로드 실패")
            return [], []

        # 2. FrameData로 변환
        frame_data_list = self._convert_to_frame_data(observation_data)
        if not frame_data_list:
            logger.warning(f"[{self.bay_id}] 처리할 프레임 데이터 없음")
            return [], []

        logger.info(f"[{self.bay_id}] {len(frame_data_list)}개 프레임 데이터 변환 완료")

        # 3. 각 프레임 투영
        projected_results = []
        for frame_data in frame_data_list:
            # 이미지 데이터 가져오기
            image_id = frame_data.image_id
            images = observation_data.get("images", {})
            image_info = images.get(image_id, {})
            image_data = image_info.get("image_data")

            projected_data = self.projector.project(frame_data, image=image_data)
            projected_results.append(projected_data)

        valid_count = sum(1 for p in projected_results if p.is_valid)
        logger.info(
            f"[{self.bay_id}] {len(projected_results)}개 중 {valid_count}개 투영 성공"
        )

        # 4. Overlap 분석
        if valid_count >= 2:
            projected_results = self.overlap_analyzer.analyze_overlaps(projected_results)

        # 5. 블록 정보 초기화
        block_infos = self._initialize_block_infos(projected_results)
        logger.info(f"[{self.bay_id}] {len(block_infos)}개 블록 정보 초기화")

        # 6. 병합 블록 정보 생성
        merged_block_infos = self._create_merged_blocks(block_infos, projected_results)
        logger.info(f"[{self.bay_id}] {len(merged_block_infos)}개 병합 블록 생성")

        return block_infos, merged_block_infos
