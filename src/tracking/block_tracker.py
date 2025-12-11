# -*- coding: utf-8 -*-
"""
Block Tracker

개별 블록의 상태를 추적하는 클래스입니다.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import Counter
import numpy as np

from .models import BlockState, AssembleState, BlockId, BBox


@dataclass
class BlockTracker:
    """개별 블록의 상태를 추적하는 클래스"""
    bbox: BBox
    contour: np.ndarray
    stage: str
    first_seen_time: datetime
    before_projected_infos: List[Dict[str, any]]

    track_id: str = field(default_factory=lambda: f"{uuid.uuid4().hex[:8]}")
    block_id: Optional[BlockId] = None
    state: BlockState = BlockState.UNIDENTIFIED
    assemble_state: AssembleState = AssembleState.PANEL
    last_seen_time: datetime = field(init=False)
    assembly_start_time: Optional[datetime] = None
    existence_score: float = 1.0
    bbox_history: List[BBox] = field(default_factory=list)
    contour_history: List[np.ndarray] = field(default_factory=list)

    # Voting history
    ship_no_counts: Dict[str, int] = field(default_factory=dict)
    block_name_counts: Dict[str, int] = field(default_factory=dict)
    stage_counts: Dict[str, int] = field(default_factory=dict)

    # 병합 추적
    merged_into_tracker_id: Optional[str] = None

    def __post_init__(self):
        self.last_seen_time = self.first_seen_time

    def update(
        self,
        bbox: BBox,
        contour: np.ndarray,
        timestamp: datetime,
        before_projected_infos: Optional[List[Dict[str, any]]] = None,
        ship_counts: Optional[Dict[str, int]] = None,
        block_counts: Optional[Dict[str, int]] = None,
        stage_list: Optional[List[str]] = None,
    ):
        """새로운 탐지 정보로 블록 상태 갱신"""
        self.bbox = bbox
        self.contour = contour
        self.last_seen_time = timestamp
        self.bbox_history.append(bbox)
        self.contour_history.append(contour)
        self.existence_score = min(1.0, self.existence_score + 0.2)

        if before_projected_infos is not None:
            self.before_projected_infos = before_projected_infos

        # Voting 업데이트
        if ship_counts is not None:
            for ship, count in ship_counts.items():
                if ship and ship.strip():
                    self.ship_no_counts[ship] = self.ship_no_counts.get(ship, 0) + count

            voted_ship = self._get_voted_value(self.ship_no_counts)
            if voted_ship and self.block_id:
                self.block_id.ship_no = voted_ship

        if block_counts is not None:
            for block, count in block_counts.items():
                if block and block.strip():
                    self.block_name_counts[block] = self.block_name_counts.get(block, 0) + count

            voted_block = self._get_voted_value(self.block_name_counts)
            if voted_block and self.block_id:
                self.block_id.block_name = voted_block

        if stage_list is not None:
            for stage in stage_list:
                if stage and stage.strip():
                    self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

            voted_stage = self._get_voted_stage()
            if voted_stage and self.block_id:
                self.block_id.stage = voted_stage
            if voted_stage:
                self.stage = voted_stage

    def _get_voted_value(self, counts: Dict[str, int]) -> Optional[str]:
        """투표로 최빈값 선택"""
        if not counts:
            return None
        counter = Counter(counts)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else None

    def _get_voted_stage(self) -> Optional[str]:
        """stage 투표 (block 키워드 우선)"""
        if not self.stage_counts:
            return None

        # "block" 키워드가 포함된 stage 우선
        block_stages = {k: v for k, v in self.stage_counts.items() if "block" in k.lower()}
        if block_stages:
            counter = Counter(block_stages)
            most_common = counter.most_common(1)
            return most_common[0][0] if most_common else None

        counter = Counter(self.stage_counts)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else None

    def promote_to_confirmed(self, block_id: BlockId):
        """UNIDENTIFIED -> CONFIRMED 승격"""
        if self.state != BlockState.UNIDENTIFIED:
            return
        self.block_id = block_id
        self.state = BlockState.CONFIRMED

    def to_lost_state(self):
        """CONFIRMED -> LOST 전환"""
        if self.state != BlockState.CONFIRMED:
            return
        self.state = BlockState.LOST

    def mark_as_exited(self):
        """EXITED 상태로 변경"""
        self.state = BlockState.EXITED

    def decrease_score(self, amount: float = 0.25, timestamp: Optional[datetime] = None):
        """존재 확률 감소"""
        if timestamp is None:
            timestamp = datetime.now()

        time_diff = timestamp - self.last_seen_time
        if time_diff.days >= 3:
            self.existence_score -= amount

    def promote_to_block(self, timestamp: datetime):
        """조립 상태를 BLOCK으로 변경"""
        if self.assemble_state == AssembleState.BLOCK:
            return
        self.assemble_state = AssembleState.BLOCK
        if self.assembly_start_time is None:
            self.assembly_start_time = timestamp

    def get_dwell_time(self) -> float:
        """체류 시간(초) 계산"""
        return (self.last_seen_time - self.first_seen_time).total_seconds()

    def initialize_block_id(self, ship_no: Optional[str], block_name: Optional[str], stage: str):
        """BlockId 초기화"""
        if self.block_id is not None:
            return

        ship_no_value = ship_no.strip() if ship_no and ship_no.strip() else ""
        block_name_value = block_name.strip() if block_name and block_name.strip() else ""

        self.block_id = BlockId(
            ship_no=ship_no_value,
            block_name=block_name_value,
            stage=stage,
        )

    def has_valid_block_id(self) -> bool:
        """BlockId가 유효한지 확인"""
        if not self.block_id:
            return False
        return bool(
            self.block_id.ship_no and self.block_id.ship_no.strip()
            and self.block_id.block_name and self.block_id.block_name.strip()
            and self.block_id.stage and self.block_id.stage.strip()
        )

    def update_state_based_on_block_id(self):
        """BlockId 유효성에 따라 상태 업데이트"""
        has_valid_id = self.has_valid_block_id()

        if self.state == BlockState.UNIDENTIFIED and has_valid_id:
            self.state = BlockState.CONFIRMED
        elif self.state == BlockState.CONFIRMED and not has_valid_id:
            self.to_lost_state()
        elif self.state == BlockState.LOST and has_valid_id:
            self.state = BlockState.CONFIRMED
