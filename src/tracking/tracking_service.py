# -*- coding: utf-8 -*-
"""
Tracking Service

블록의 이동을 추적하고 이벤트를 기록하는 메인 서비스입니다.
Fusion 결과 (Pickle/CSV)를 읽어서 처리합니다.
"""

import pickle
import gzip
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

from .models import BlockState, EventType, AssembleState, BlockId, BBox, CSVDetection
from .block_tracker import BlockTracker
from .event_logger import EventLogger

logger = logging.getLogger(__name__)


class TrackingService:
    """
    블록 이동을 추적하고 이벤트를 기록하는 서비스

    Redis pub-sub 대신 Pickle/CSV 파일 기반으로 데이터를 교환합니다.
    """

    def __init__(
        self,
        bay_id: str,
        base_path: str = "/opt/airflow/data",
        iou_threshold: float = 0.3,
        existence_threshold: float = 0.0,
    ):
        """
        Args:
            bay_id: Bay 식별자
            base_path: 데이터 저장 기본 경로
            iou_threshold: IOU 매칭 임계값
            existence_threshold: 존재 확률 임계값
        """
        self.bay_id = bay_id
        self.base_path = Path(base_path)
        self.iou_threshold = iou_threshold
        self.existence_threshold = existence_threshold

        # 트래커 관리
        self.trackers: Dict[str, BlockTracker] = {}

        # 이벤트 로거 초기화
        event_log_path = self.base_path / "tracking" / bay_id / "tracking_events.csv"
        self.event_logger = EventLogger(event_log_path)
        self.event_logger.initialize()

        logger.info(f"[{bay_id}] TrackingService 초기화 완료")

    def load_fusion_result(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Fusion 결과를 Pickle 파일에서 로드

        Args:
            batch_id: 배치 식별자

        Returns:
            로드된 데이터 또는 None
        """
        result_path = (
            self.base_path / "fusion" / self.bay_id / batch_id / "result.pkl"
        )

        try:
            if not result_path.exists():
                logger.error(f"[{self.bay_id}] Fusion 결과 파일 없음: {result_path}")
                return None

            try:
                with gzip.open(result_path, "rb") as f:
                    data = pickle.load(f)
            except gzip.BadGzipFile:
                with open(result_path, "rb") as f:
                    data = pickle.load(f)

            logger.info(f"[{self.bay_id}] Fusion 결과 로드 완료: {batch_id}")
            return data

        except Exception as e:
            logger.error(f"[{self.bay_id}] Fusion 결과 로드 실패: {e}")
            return None

    def _convert_to_detections(
        self, fusion_data: Dict[str, Any], batch_id: str
    ) -> List[CSVDetection]:
        """
        Fusion 데이터를 CSVDetection 리스트로 변환
        """
        detections = []
        merged_block_infos = fusion_data.get("merged_block_infos", [])

        for info in merged_block_infos:
            # bbox_coordinate를 BBox로 변환
            bbox_coord = info.get("bbox_coordinate", [])
            if isinstance(bbox_coord, list):
                bbox_coord = np.array(bbox_coord)

            bbox = BBox(bbox_coord=bbox_coord) if bbox_coord.size > 0 else None

            # contour (warped_mask_contour에서)
            contour_data = info.get("warped_mask_contour", [])
            contour = np.array(contour_data) if contour_data else None

            # timestamp 파싱
            timestamp_str = info.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now()

            detection = CSVDetection(
                bbox=bbox,
                contour=contour,
                timestamps=[timestamp],
                vision_sensor_ids=info.get("vision_sensor_id", []),
                top1_classes=info.get("top1_class", []),
                stage_labels=info.get("stage_labels", []),
                image_paths=info.get("image_paths", []),
                original_bbox_coordinates=[
                    np.array(c) if isinstance(c, list) else c
                    for c in info.get("original_bbox_coordinates", [])
                ],
                before_projected_bboxes_xywh=info.get("before_projected_bboxes_xywh", []),
                matched_ships_counts=info.get("matched_ships_counts", {}),
                matched_blocks_counts=info.get("matched_blocks_counts", {}),
                voted_ships=[info.get("voted_ship", "")],
                voted_blocks=[info.get("voted_block", "")],
                batch_ids=[batch_id],
                representate_ship=info.get("voted_ship", ""),
                representate_block=info.get("voted_block", ""),
                representate_stage=info.get("stage_labels", [""])[0] if info.get("stage_labels") else "",
            )
            detections.append(detection)

        return detections

    def _match_detection_to_tracker(
        self, detection: CSVDetection
    ) -> Optional[BlockTracker]:
        """
        Detection을 기존 트래커와 매칭
        """
        if detection.bbox is None:
            return None

        best_match: Optional[BlockTracker] = None
        best_iou = self.iou_threshold

        for tracker in self.trackers.values():
            if tracker.state == BlockState.EXITED:
                continue
            if tracker.merged_into_tracker_id is not None:
                continue

            iou = tracker.bbox.iou(detection.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = tracker

        return best_match

    def _create_new_tracker(
        self,
        detection: CSVDetection,
        timestamp: datetime,
    ) -> BlockTracker:
        """
        새로운 트래커 생성
        """
        bbox = detection.bbox
        contour = detection.contour if detection.contour is not None else np.array([])
        stage = detection.representate_stage or ""

        # before_projected_infos 구성
        before_projected_infos = []
        for i, camera_id in enumerate(detection.vision_sensor_ids):
            info = {"camera_name": camera_id}
            if i < len(detection.image_paths):
                info["image_path"] = detection.image_paths[i]
            if i < len(detection.before_projected_bboxes_xywh):
                info["bbox_xywh"] = detection.before_projected_bboxes_xywh[i]
            before_projected_infos.append(info)

        tracker = BlockTracker(
            bbox=bbox,
            contour=contour,
            stage=stage,
            first_seen_time=timestamp,
            before_projected_infos=before_projected_infos,
        )

        # BlockId 초기화
        tracker.initialize_block_id(
            ship_no=detection.representate_ship,
            block_name=detection.representate_block,
            stage=stage,
        )

        # Voting 초기화
        if detection.matched_ships_counts:
            tracker.ship_no_counts = detection.matched_ships_counts.copy()
        if detection.matched_blocks_counts:
            tracker.block_name_counts = detection.matched_blocks_counts.copy()

        self.trackers[tracker.track_id] = tracker
        return tracker

    def _log_entry_event(self, tracker: BlockTracker, timestamp: datetime):
        """ENTRY 이벤트 기록"""
        self.event_logger.save_event(
            timestamp=timestamp,
            track_id=tracker.track_id,
            event_type=EventType.ENTRY,
            bay=self.bay_id,
            subbay="",
            ship_no=tracker.block_id.ship_no if tracker.block_id else "",
            block_name=tracker.block_id.block_name if tracker.block_id else "",
            stage=tracker.stage,
            description="블록 반입(배열)",
        )

    def _log_assembly_start_event(self, tracker: BlockTracker, timestamp: datetime):
        """START_ASSEMBLY 이벤트 기록"""
        self.event_logger.save_event(
            timestamp=timestamp,
            track_id=tracker.track_id,
            event_type=EventType.START_ASSEMBLY,
            bay=self.bay_id,
            subbay="",
            ship_no=tracker.block_id.ship_no if tracker.block_id else "",
            block_name=tracker.block_id.block_name if tracker.block_id else "",
            stage=tracker.stage,
            description="조립 착수",
        )

    def _log_exit_event(self, tracker: BlockTracker, timestamp: datetime):
        """EXIT 이벤트 기록"""
        self.event_logger.save_event(
            timestamp=timestamp,
            track_id=tracker.track_id,
            event_type=EventType.EXIT,
            bay=self.bay_id,
            subbay="",
            ship_no=tracker.block_id.ship_no if tracker.block_id else "",
            block_name=tracker.block_id.block_name if tracker.block_id else "",
            stage=tracker.stage,
            description="블록 반출",
        )

    def process_timestep(
        self, timestamp: datetime, batch_id: str
    ) -> Dict[str, Any]:
        """
        단일 타임스텝 처리

        Args:
            timestamp: 처리 시간
            batch_id: 배치 식별자

        Returns:
            처리 결과 요약
        """
        logger.info(f"[{self.bay_id}] Timestep 처리: {batch_id}")

        # 1. Fusion 결과 로드
        fusion_data = self.load_fusion_result(batch_id)
        if not fusion_data:
            logger.warning(f"[{self.bay_id}] Fusion 데이터 없음, 건너뜀")
            return {"success": False, "error": "No fusion data"}

        # 2. Detection으로 변환
        detections = self._convert_to_detections(fusion_data, batch_id)
        logger.info(f"[{self.bay_id}] {len(detections)}개 detection 변환 완료")

        # 3. 매칭 및 업데이트
        matched_count = 0
        new_count = 0
        matched_tracker_ids = set()

        for detection in detections:
            if detection.bbox is None:
                continue

            tracker = self._match_detection_to_tracker(detection)

            if tracker:
                # 기존 트래커 업데이트
                before_projected_infos = []
                for i, camera_id in enumerate(detection.vision_sensor_ids):
                    info = {"camera_name": camera_id}
                    if i < len(detection.image_paths):
                        info["image_path"] = detection.image_paths[i]
                    if i < len(detection.before_projected_bboxes_xywh):
                        info["bbox_xywh"] = detection.before_projected_bboxes_xywh[i]
                    before_projected_infos.append(info)

                tracker.update(
                    bbox=detection.bbox,
                    contour=detection.contour if detection.contour is not None else np.array([]),
                    timestamp=timestamp,
                    before_projected_infos=before_projected_infos,
                    ship_counts=detection.matched_ships_counts,
                    block_counts=detection.matched_blocks_counts,
                    stage_list=detection.stage_labels,
                )

                # 조립 착수 체크
                if tracker.assemble_state == AssembleState.PANEL:
                    stage = detection.representate_stage or ""
                    if "block" in stage.lower():
                        tracker.promote_to_block(timestamp)
                        self._log_assembly_start_event(tracker, timestamp)

                tracker.update_state_based_on_block_id()
                matched_tracker_ids.add(tracker.track_id)
                matched_count += 1

            else:
                # 새 트래커 생성
                new_tracker = self._create_new_tracker(detection, timestamp)
                self._log_entry_event(new_tracker, timestamp)
                new_count += 1

        # 4. 매칭되지 않은 트래커 처리
        for track_id, tracker in list(self.trackers.items()):
            if track_id not in matched_tracker_ids:
                if tracker.state != BlockState.EXITED:
                    tracker.decrease_score(0.25, timestamp)

                    if tracker.existence_score <= self.existence_threshold:
                        tracker.mark_as_exited()
                        self._log_exit_event(tracker, timestamp)

        # 5. 결과 요약
        active_trackers = sum(
            1
            for t in self.trackers.values()
            if t.state != BlockState.EXITED and t.merged_into_tracker_id is None
        )

        result = {
            "success": True,
            "batch_id": batch_id,
            "timestamp": timestamp.isoformat(),
            "detections_count": len(detections),
            "matched_count": matched_count,
            "new_count": new_count,
            "active_trackers": active_trackers,
            "total_trackers": len(self.trackers),
        }

        logger.info(
            f"[{self.bay_id}] Timestep 완료: matched={matched_count}, new={new_count}, active={active_trackers}"
        )

        return result

    def get_active_trackers(self) -> List[BlockTracker]:
        """활성 트래커 목록 반환"""
        return [
            t
            for t in self.trackers.values()
            if t.state != BlockState.EXITED and t.merged_into_tracker_id is None
        ]

    def get_tracker_by_id(self, track_id: str) -> Optional[BlockTracker]:
        """특정 트래커 조회"""
        return self.trackers.get(track_id)

    def save_trackers_state(self, output_path: Optional[Path] = None) -> Path:
        """
        현재 트래커 상태를 Pickle로 저장

        Args:
            output_path: 저장 경로 (None이면 기본 경로 사용)

        Returns:
            저장된 파일 경로
        """
        if output_path is None:
            output_path = self.base_path / "tracking" / self.bay_id / "trackers_state.pkl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 직렬화 가능한 형태로 변환
        state_data = {
            "bay_id": self.bay_id,
            "saved_at": datetime.now().isoformat(),
            "trackers": {},
        }

        for track_id, tracker in self.trackers.items():
            tracker_data = {
                "track_id": tracker.track_id,
                "state": tracker.state.name,
                "assemble_state": tracker.assemble_state.name,
                "first_seen_time": tracker.first_seen_time.isoformat(),
                "last_seen_time": tracker.last_seen_time.isoformat(),
                "existence_score": tracker.existence_score,
                "stage": tracker.stage,
                "bbox_coord": tracker.bbox.bbox_coord.tolist() if tracker.bbox else [],
            }

            if tracker.block_id:
                tracker_data["block_id"] = {
                    "ship_no": tracker.block_id.ship_no,
                    "block_name": tracker.block_id.block_name,
                    "stage": tracker.block_id.stage,
                }

            state_data["trackers"][track_id] = tracker_data

        with gzip.open(output_path, "wb") as f:
            pickle.dump(state_data, f)

        logger.info(f"[{self.bay_id}] 트래커 상태 저장 완료: {output_path}")
        return output_path

    def load_trackers_state(self, input_path: Optional[Path] = None) -> bool:
        """
        저장된 트래커 상태 로드

        Args:
            input_path: 로드할 파일 경로 (None이면 기본 경로 사용)

        Returns:
            로드 성공 여부
        """
        if input_path is None:
            input_path = self.base_path / "tracking" / self.bay_id / "trackers_state.pkl"

        if not input_path.exists():
            logger.warning(f"[{self.bay_id}] 트래커 상태 파일 없음: {input_path}")
            return False

        try:
            with gzip.open(input_path, "rb") as f:
                state_data = pickle.load(f)

            # 트래커 복원
            for track_id, tracker_data in state_data.get("trackers", {}).items():
                bbox_coord = np.array(tracker_data.get("bbox_coord", []))
                bbox = BBox(bbox_coord=bbox_coord) if bbox_coord.size > 0 else BBox(bbox_coord=np.zeros((4, 2)))

                tracker = BlockTracker(
                    bbox=bbox,
                    contour=np.array([]),
                    stage=tracker_data.get("stage", ""),
                    first_seen_time=datetime.fromisoformat(tracker_data["first_seen_time"]),
                    before_projected_infos=[],
                )
                tracker.track_id = tracker_data["track_id"]
                tracker.state = BlockState[tracker_data["state"]]
                tracker.assemble_state = AssembleState[tracker_data["assemble_state"]]
                tracker.last_seen_time = datetime.fromisoformat(tracker_data["last_seen_time"])
                tracker.existence_score = tracker_data.get("existence_score", 1.0)

                if "block_id" in tracker_data:
                    bid = tracker_data["block_id"]
                    tracker.block_id = BlockId(
                        ship_no=bid.get("ship_no", ""),
                        block_name=bid.get("block_name", ""),
                        stage=bid.get("stage", ""),
                    )

                self.trackers[track_id] = tracker

            logger.info(
                f"[{self.bay_id}] 트래커 상태 로드 완료: {len(self.trackers)}개 트래커"
            )
            return True

        except Exception as e:
            logger.error(f"[{self.bay_id}] 트래커 상태 로드 실패: {e}")
            return False
