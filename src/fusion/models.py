# -*- coding: utf-8 -*-
"""
Fusion 모듈 데이터 모델

다중 카메라 데이터 융합을 위한 데이터 구조를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class FrameData:
    """
    단일 카메라 프레임의 모든 정보를 담는 DTO

    Observation 모듈에서 전달받은 데이터를 Fusion 처리를 위해 구조화합니다.
    """
    image_id: str
    camera_name: str
    image_path: str
    captured_at: datetime
    image_shape: Tuple[int, int]
    detections: List[Dict[str, Any]] = field(default_factory=list)
    boundary_masks: List[Dict[str, Any]] = field(default_factory=list)
    assembly_classifications: List[Dict[str, Any]] = field(default_factory=list)
    pinjig_masks: List[np.ndarray] = field(default_factory=list)
    pinjig_classifications: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectedData:
    """
    단일 카메라의 데이터를 절대 좌표계(Bird's-eye view)로 투영한 결과물
    """
    image_id: str
    camera_name: str
    image_path: str
    captured_at: datetime
    warped_image: Optional[np.ndarray]
    warped_masks: List[np.ndarray]
    warped_pinjig_masks: List[np.ndarray]
    projected_boxes: List[np.ndarray]
    projected_assembly_boxes: List[np.ndarray]
    projected_assembly_labels: List[Dict[str, Any]]
    extent: List[float]  # [min_x, max_x, max_y, min_y]
    clip_polygon: np.ndarray
    projected_boxes_labels: List[str] = field(default_factory=list)
    before_projected_bboxes_xywh: List[List[float]] = field(default_factory=list)
    is_valid: bool = True
    merged_boxes: List[np.ndarray] = field(default_factory=list)
    mask_merge_mapping: Dict[int, str] = field(default_factory=dict)
    merge_group_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MergedBlockInfo:
    """
    병합된 블록 정보

    여러 카메라에서 감지된 동일 블록의 정보를 통합합니다.
    """
    vision_sensor_id: List[str]
    top1_class: List[str]
    bbox_coordinate: np.ndarray
    warped_mask_shape: Optional[Tuple[int, int]]
    warped_mask_contour: List[float]
    stage_labels: List[str]
    image_path: str
    image_paths: List[str]
    original_bbox_coordinates: List[np.ndarray]
    before_projected_bboxes_xywh: List[List[float]]
    matched_text_recognitions: List[Dict[str, Any]]
    matched_object_counts: Dict[str, int]
    matched_ships_counts: Dict[str, int]
    matched_blocks_counts: Dict[str, int]
    source_infos: List[Dict[str, Any]]
    is_merged: int
    merged_block_idx: List[int]
    voted_ship: str
    voted_block: str
    block_idx_in_layout: str
    merge_group_id: str
    batch_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "vision_sensor_id": self.vision_sensor_id,
            "top1_class": self.top1_class,
            "bbox_coordinate": self.bbox_coordinate.tolist() if isinstance(self.bbox_coordinate, np.ndarray) else self.bbox_coordinate,
            "warped_mask_shape": list(self.warped_mask_shape) if self.warped_mask_shape else [],
            "warped_mask_contour": self.warped_mask_contour,
            "stage_labels": self.stage_labels,
            "image_path": self.image_path,
            "image_paths": self.image_paths,
            "original_bbox_coordinates": [
                c.tolist() if isinstance(c, np.ndarray) else c
                for c in self.original_bbox_coordinates
            ],
            "before_projected_bboxes_xywh": self.before_projected_bboxes_xywh,
            "matched_text_recognitions": self.matched_text_recognitions,
            "matched_object_counts": self.matched_object_counts,
            "matched_ships_counts": self.matched_ships_counts,
            "matched_blocks_counts": self.matched_blocks_counts,
            "source_infos": self.source_infos,
            "is_merged": self.is_merged,
            "merged_block_idx": self.merged_block_idx,
            "voted_ship": self.voted_ship,
            "voted_block": self.voted_block,
            "block_idx_in_layout": self.block_idx_in_layout,
            "merge_group_id": self.merge_group_id,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergedBlockInfo":
        """딕셔너리에서 생성"""
        bbox_coord = data.get("bbox_coordinate", [])
        if isinstance(bbox_coord, list):
            bbox_coord = np.array(bbox_coord)

        original_bboxes = data.get("original_bbox_coordinates", [])
        original_bboxes = [
            np.array(c) if isinstance(c, list) else c
            for c in original_bboxes
        ]

        return cls(
            vision_sensor_id=data.get("vision_sensor_id", []),
            top1_class=data.get("top1_class", []),
            bbox_coordinate=bbox_coord,
            warped_mask_shape=tuple(data.get("warped_mask_shape", [])) if data.get("warped_mask_shape") else None,
            warped_mask_contour=data.get("warped_mask_contour", []),
            stage_labels=data.get("stage_labels", []),
            image_path=data.get("image_path", ""),
            image_paths=data.get("image_paths", []),
            original_bbox_coordinates=original_bboxes,
            before_projected_bboxes_xywh=data.get("before_projected_bboxes_xywh", []),
            matched_text_recognitions=data.get("matched_text_recognitions", []),
            matched_object_counts=data.get("matched_object_counts", {}),
            matched_ships_counts=data.get("matched_ships_counts", {}),
            matched_blocks_counts=data.get("matched_blocks_counts", {}),
            source_infos=data.get("source_infos", []),
            is_merged=data.get("is_merged", 0),
            merged_block_idx=data.get("merged_block_idx", []),
            voted_ship=data.get("voted_ship", ""),
            voted_block=data.get("voted_block", ""),
            block_idx_in_layout=data.get("block_idx_in_layout", ""),
            merge_group_id=data.get("merge_group_id", ""),
            batch_id=data.get("batch_id", ""),
            timestamp=data.get("timestamp", ""),
        )
