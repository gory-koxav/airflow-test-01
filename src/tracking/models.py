# -*- coding: utf-8 -*-
"""
Tracking 모듈 데이터 모델

블록 추적을 위한 데이터 구조를 정의합니다.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np


class BlockState(Enum):
    """블록의 생애주기 상태"""
    UNIDENTIFIED = auto()  # ID-미확인
    CONFIRMED = auto()     # ID-확인
    LOST = auto()          # ID-소실 / 추적
    EXITED = auto()        # 반출


class EventType(Enum):
    """추적 이벤트 유형"""
    ENTRY = auto()              # 반입(배열)
    START_ASSEMBLY = auto()     # 조립 착수
    EXIT = auto()               # 반출
    SWAP_ENTRY = auto()         # 교체 반입
    SWAP_EXIT = auto()          # 교체 반출
    TEMP_STORAGE_EXIT = auto()  # 임시보관 반출
    STATE_CHANGE = auto()       # 상태 변화
    MERGE = auto()              # 트래커 병합


class AssembleState(Enum):
    """블록의 조립 상태"""
    PANEL = auto()  # 판넬 상태
    BLOCK = auto()  # 블록 상태


@dataclass
class BlockId:
    """블록 식별자"""
    ship_no: str
    block_name: str
    stage: str

    def __str__(self) -> str:
        return f"S:{self.ship_no}-B:{self.block_name}"

    def __repr__(self) -> str:
        return f"BlockId(ship_no='{self.ship_no}', block_name='{self.block_name}', stage='{self.stage}')"

    @classmethod
    def from_string(cls, block_id_str: str, stage: str = "") -> "BlockId":
        """문자열에서 BlockId 생성"""
        try:
            if not block_id_str or block_id_str == "N/A":
                return cls(ship_no="", block_name="", stage=stage)

            parts = block_id_str.split("-")
            if len(parts) >= 2:
                ship_part = parts[0]
                block_part = parts[1]

                ship_no = ship_part[2:] if ship_part.startswith("S:") else ship_part
                block_name = block_part[2:] if block_part.startswith("B:") else block_part

                return cls(ship_no=ship_no, block_name=block_name, stage=stage)
            else:
                return cls(ship_no="", block_name="", stage=stage)
        except Exception:
            return cls(ship_no="", block_name="", stage=stage)


@dataclass
class BBox:
    """경계 상자 (4개 corner points)"""
    bbox_coord: np.ndarray  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float = 1.0

    def center(self) -> Tuple[float, float]:
        """중심점 좌표"""
        try:
            if self.bbox_coord.shape != (4, 2):
                return (0.0, 0.0)
            cx = np.mean(self.bbox_coord[:, 0])
            cy = np.mean(self.bbox_coord[:, 1])
            return (float(cx), float(cy))
        except Exception:
            return (0.0, 0.0)

    def area(self) -> float:
        """면적 계산 (Shoelace formula)"""
        try:
            if self.bbox_coord.shape != (4, 2):
                return 0.0
            n = len(self.bbox_coord)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += self.bbox_coord[i][0] * self.bbox_coord[j][1]
                area -= self.bbox_coord[j][0] * self.bbox_coord[i][1]
            return abs(area) / 2.0
        except Exception:
            return 0.0

    def iou(self, other: "BBox") -> float:
        """IOU 계산 (간략화된 버전)"""
        try:
            # 간단히 AABB로 근사
            min_x1, max_x1 = np.min(self.bbox_coord[:, 0]), np.max(self.bbox_coord[:, 0])
            min_y1, max_y1 = np.min(self.bbox_coord[:, 1]), np.max(self.bbox_coord[:, 1])
            min_x2, max_x2 = np.min(other.bbox_coord[:, 0]), np.max(other.bbox_coord[:, 0])
            min_y2, max_y2 = np.min(other.bbox_coord[:, 1]), np.max(other.bbox_coord[:, 1])

            inter_x1 = max(min_x1, min_x2)
            inter_y1 = max(min_y1, min_y2)
            inter_x2 = min(max_x1, max_x2)
            inter_y2 = min(max_y1, max_y2)

            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                return 0.0

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
            area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
            union_area = area1 + area2 - inter_area

            return float(inter_area / union_area) if union_area > 0 else 0.0
        except Exception:
            return 0.0


@dataclass
class TrackingEvent:
    """추적 이벤트"""
    timestamp: datetime
    event_type: EventType
    track_id: str
    bay: str = ""
    subbay: str = ""
    ship_no: str = ""
    block_name: str = ""
    stage: str = ""
    stage_code: str = ""
    description: str = ""
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    delete_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "time": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": self.track_id,
            "bay": self.bay,
            "subbay": self.subbay,
            "event_type": self.event_type.name,
            "ship": self.ship_no,
            "block": self.block_name,
            "stage": self.stage,
            "stage_code": self.stage_code,
            "description": self.description,
            "is_deleted": str(self.is_deleted),
            "deleted_at": self.deleted_at.strftime("%Y-%m-%d %H:%M:%S") if self.deleted_at else "",
            "delete_reason": self.delete_reason,
        }


@dataclass
class CSVDetection:
    """CSV 데이터 기반 탐지 결과"""
    bbox: Optional[BBox]
    contour: Optional[np.ndarray] = None
    timestamps: List[datetime] = field(default_factory=list)
    vision_sensor_ids: List[str] = field(default_factory=list)
    top1_classes: List[str] = field(default_factory=list)
    stage_labels: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    original_bbox_coordinates: List[np.ndarray] = field(default_factory=list)
    before_projected_bboxes_xywh: List[List[float]] = field(default_factory=list)
    matched_ships_counts: Dict[str, int] = field(default_factory=dict)
    matched_blocks_counts: Dict[str, int] = field(default_factory=dict)
    voted_ships: List[str] = field(default_factory=list)
    voted_blocks: List[str] = field(default_factory=list)
    batch_ids: List[str] = field(default_factory=list)
    representate_ship: str = ""
    representate_block: str = ""
    representate_stage: str = ""
