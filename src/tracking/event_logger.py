# -*- coding: utf-8 -*-
"""
Event Logger

추적 이벤트를 CSV 파일로 기록합니다.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging

from .models import EventType, TrackingEvent

logger = logging.getLogger(__name__)


class EventLogger:
    """
    추적 이벤트를 CSV 파일로 기록하는 클래스

    Redis 대신 CSV 파일 기반으로 이벤트를 저장합니다.
    """

    HEADERS = [
        "time",
        "track_id",
        "bay",
        "subbay",
        "event_type",
        "ship",
        "block",
        "stage",
        "stage_code",
        "description",
        "is_deleted",
        "deleted_at",
        "delete_reason",
    ]

    def __init__(self, csv_file_path: Path):
        """
        Args:
            csv_file_path: CSV 파일 경로
        """
        self.csv_file_path = csv_file_path
        self._initialized = False

    def initialize(self) -> bool:
        """CSV 파일 초기화"""
        try:
            self.csv_file_path.parent.mkdir(parents=True, exist_ok=True)

            if not self._validate_headers():
                self._recreate_csv()
            elif not self.csv_file_path.exists():
                with open(self.csv_file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.HEADERS)
                logger.info(f"CSV 파일 초기화 완료: {self.csv_file_path}")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"CSV 초기화 실패: {e}")
            self._initialized = False
            return False

    def _validate_headers(self) -> bool:
        """CSV 헤더 검증"""
        if not self.csv_file_path.exists():
            return True

        try:
            with open(self.csv_file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)
                if existing_headers is None:
                    return False
                return existing_headers == self.HEADERS
        except Exception:
            return True

    def _recreate_csv(self):
        """CSV 파일 재생성"""
        try:
            if self.csv_file_path.exists():
                self.csv_file_path.unlink()

            with open(self.csv_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

            logger.info(f"CSV 파일 재생성 완료: {self.csv_file_path}")
        except Exception as e:
            logger.error(f"CSV 재생성 실패: {e}")

    def log_event(self, event: TrackingEvent) -> bool:
        """
        이벤트 기록

        Args:
            event: 기록할 이벤트

        Returns:
            성공 여부
        """
        # STATE_CHANGE 이벤트는 기록하지 않음
        if event.event_type == EventType.STATE_CHANGE:
            return True

        if not self._initialized:
            if not self.initialize():
                return False

        try:
            row_data = [
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                event.track_id,
                event.bay,
                event.subbay,
                event.event_type.name,
                event.ship_no,
                event.block_name,
                event.stage,
                event.stage_code,
                event.description,
                "False",
                "",
                "",
            ]

            with open(self.csv_file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

            logger.debug(
                f"이벤트 기록: {event.event_type.name} - {event.track_id}"
            )
            return True

        except Exception as e:
            logger.error(f"이벤트 기록 실패: {e}")
            return False

    def save_event(
        self,
        timestamp: datetime,
        track_id: str,
        event_type: EventType,
        bay: str = "",
        subbay: str = "",
        ship_no: str = "",
        block_name: str = "",
        stage: str = "",
        stage_code: str = "",
        description: str = "",
    ) -> bool:
        """
        이벤트 저장 (편의 메서드)
        """
        event = TrackingEvent(
            timestamp=timestamp,
            event_type=event_type,
            track_id=track_id,
            bay=bay,
            subbay=subbay,
            ship_no=ship_no,
            block_name=block_name,
            stage=stage,
            stage_code=stage_code,
            description=description,
        )
        return self.log_event(event)

    def soft_delete_by_track_id(
        self,
        track_id: str,
        deleted_at: datetime,
        delete_reason: str,
    ) -> tuple:
        """
        특정 track_id의 이벤트들을 소프트 삭제

        Args:
            track_id: 삭제할 트래커 ID
            deleted_at: 삭제 시간
            delete_reason: 삭제 이유

        Returns:
            (성공 여부, 삭제된 레코드 수)
        """
        if not self.csv_file_path.exists():
            return (True, 0)

        try:
            rows = []
            deleted_count = 0

            with open(self.csv_file_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers is None:
                    return (True, 0)

                rows.append(headers)

                track_id_idx = headers.index("track_id")
                is_deleted_idx = headers.index("is_deleted")
                deleted_at_idx = headers.index("deleted_at")
                delete_reason_idx = headers.index("delete_reason")

                for row in reader:
                    if len(row) <= max(track_id_idx, is_deleted_idx, deleted_at_idx, delete_reason_idx):
                        rows.append(row)
                        continue

                    if row[track_id_idx] == track_id and row[is_deleted_idx] != "True":
                        row[is_deleted_idx] = "True"
                        row[deleted_at_idx] = deleted_at.strftime("%Y-%m-%d %H:%M:%S")
                        row[delete_reason_idx] = delete_reason
                        deleted_count += 1

                    rows.append(row)

            if deleted_count > 0:
                with open(self.csv_file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)

                logger.info(
                    f"소프트 삭제 완료: track_id={track_id}, 삭제 수={deleted_count}"
                )

            return (True, deleted_count)

        except Exception as e:
            logger.error(f"소프트 삭제 실패: {e}")
            return (False, 0)

    def is_available(self) -> bool:
        """CSV 파일 접근 가능 여부"""
        if not self._initialized:
            return False

        try:
            if self.csv_file_path.exists():
                with open(self.csv_file_path, "a", encoding="utf-8"):
                    pass
                return True
            return True
        except Exception:
            return False

    def close(self):
        """리소스 정리 (CSV는 작업마다 열고 닫으므로 no-op)"""
        pass
