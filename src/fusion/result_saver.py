# -*- coding: utf-8 -*-
"""
Fusion Result Saver

Fusion 결과를 Pickle 및 CSV 파일로 저장합니다.
"""

import csv
import json
import pickle
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusionSaveResult:
    """저장 결과 데이터 모델"""
    success: bool
    pickle_path: str
    csv_path: str
    batch_id: str
    bay_id: str
    merged_block_count: int
    error_message: Optional[str] = None


class FusionResultSaver:
    """
    Fusion 결과를 Pickle 및 CSV 파일로 저장합니다.

    저장 경로 구조:
    - Pickle: {base_path}/fusion/{bay_id}/{batch_id}/result.pkl
    - CSV: {base_path}/fusion/{bay_id}/merged_block_infos.csv
    """

    # CSV 헤더 정의
    MERGED_BLOCK_INFOS_HEADERS = [
        "batch_id",
        "vision_sensor_id",
        "top1_class",
        "bbox_coordinate",
        "warped_mask_shape",
        "warped_mask_contour",
        "stage_labels",
        "image_path",
        "image_paths",
        "original_bbox_coordinates",
        "before_projected_bboxes_xywh",
        "matched_text_recognitions",
        "matched_object_counts",
        "matched_ships_counts",
        "matched_blocks_counts",
        "source_infos",
        "is_merged",
        "merged_block_idx",
        "voted_ship",
        "voted_block",
        "block_idx_in_layout",
        "merge_group_id",
        "timestamp",
    ]

    def __init__(
        self,
        base_path: str = "/opt/airflow/data",
        compress: bool = True,
        compression_level: int = 6,
        retention_months: int = 3,
    ):
        """
        Args:
            base_path: 데이터 저장 기본 경로
            compress: Pickle 압축 여부
            compression_level: gzip 압축 레벨 (1-9)
            retention_months: CSV 데이터 보관 기간 (월)
        """
        self.base_path = Path(base_path)
        self.compress = compress
        self.compression_level = compression_level
        self.retention_months = retention_months

    def _serialize_data(self, data: Any) -> str:
        """복잡한 데이터 타입을 JSON 문자열로 직렬화"""
        try:
            if isinstance(data, np.ndarray):
                return json.dumps(data.tolist())
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
                return json.dumps([arr.tolist() for arr in data])
            else:
                return json.dumps(data, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"데이터 직렬화 실패: {e}")
            return ""

    def _validate_csv_headers(self, file_path: Path) -> bool:
        """CSV 파일 헤더 검증"""
        try:
            if not file_path.exists():
                return True

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                file_headers = next(reader, [])
                return file_headers == self.MERGED_BLOCK_INFOS_HEADERS

        except Exception as e:
            logger.error(f"CSV 헤더 검증 실패: {e}")
            return True

    def _cleanup_old_csv_records(self, file_path: Path) -> None:
        """오래된 CSV 레코드 정리"""
        try:
            if not file_path.exists():
                return

            if file_path.stat().st_size < 100:
                return

            cutoff_date = datetime.now() - timedelta(days=self.retention_months * 30)
            cutoff_date_str = cutoff_date.strftime("%Y%m%d")

            # CSV 파일 읽기
            rows_to_keep = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    rows_to_keep.append(headers)

                for row in reader:
                    if len(row) > 0:
                        batch_id = row[0]
                        # batch_id에서 날짜 추출 (YYYYMMDD-HHMMSS 형식)
                        date_part = batch_id.split("-")[0][:8] if "-" in batch_id else batch_id[:8]
                        if date_part >= cutoff_date_str:
                            rows_to_keep.append(row)

            original_count = sum(1 for _ in open(file_path, "r", encoding="utf-8")) - 1
            deleted_count = original_count - (len(rows_to_keep) - 1)

            if deleted_count > 0:
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows_to_keep)
                logger.info(
                    f"CSV 정리 완료: {deleted_count}개 레코드 삭제, {len(rows_to_keep) - 1}개 보관"
                )

        except Exception as e:
            logger.error(f"CSV 정리 실패: {e}")

    def save_results(
        self,
        merged_block_infos: List[Dict[str, Any]],
        batch_id: str,
        bay_id: str,
    ) -> FusionSaveResult:
        """
        Fusion 결과를 Pickle 및 CSV로 저장

        Args:
            merged_block_infos: 병합된 블록 정보 리스트
            batch_id: 배치 식별자
            bay_id: Bay 식별자

        Returns:
            FusionSaveResult: 저장 결과
        """
        try:
            # Pickle 저장
            pickle_path = self._save_pickle(merged_block_infos, batch_id, bay_id)

            # CSV 저장
            csv_path = self._save_csv(merged_block_infos, batch_id, bay_id)

            logger.info(f"[{bay_id}] Fusion 결과 저장 완료: {batch_id}")

            return FusionSaveResult(
                success=True,
                pickle_path=str(pickle_path),
                csv_path=str(csv_path),
                batch_id=batch_id,
                bay_id=bay_id,
                merged_block_count=len(merged_block_infos),
            )

        except Exception as e:
            logger.error(f"[{bay_id}] Fusion 결과 저장 실패: {e}")
            return FusionSaveResult(
                success=False,
                pickle_path="",
                csv_path="",
                batch_id=batch_id,
                bay_id=bay_id,
                merged_block_count=0,
                error_message=str(e),
            )

    def _save_pickle(
        self,
        merged_block_infos: List[Dict[str, Any]],
        batch_id: str,
        bay_id: str,
    ) -> Path:
        """Pickle 파일 저장"""
        save_dir = self.base_path / "fusion" / bay_id / batch_id
        save_dir.mkdir(parents=True, exist_ok=True)

        result_path = save_dir / "result.pkl"

        # numpy 배열을 리스트로 변환하여 저장
        serializable_data = []
        for info in merged_block_infos:
            serializable_info = {}
            for key, value in info.items():
                if key.startswith("_"):
                    continue  # 내부 데이터 제외
                if isinstance(value, np.ndarray):
                    serializable_info[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_info[key] = [v.tolist() for v in value]
                else:
                    serializable_info[key] = value
            serializable_data.append(serializable_info)

        save_data = {
            "batch_id": batch_id,
            "bay_id": bay_id,
            "saved_at": datetime.now().isoformat(),
            "merged_block_infos": serializable_data,
            "summary": {
                "total_merged_blocks": len(merged_block_infos),
            },
        }

        if self.compress:
            with gzip.open(result_path, "wb", compresslevel=self.compression_level) as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(result_path, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"[{bay_id}] Pickle 저장 완료: {result_path}")
        return result_path

    def _save_csv(
        self,
        merged_block_infos: List[Dict[str, Any]],
        batch_id: str,
        bay_id: str,
    ) -> Path:
        """CSV 파일 저장"""
        csv_dir = self.base_path / "fusion" / bay_id
        csv_dir.mkdir(parents=True, exist_ok=True)

        csv_path = csv_dir / "merged_block_infos.csv"

        # 헤더 검증 및 필요시 재생성
        write_header = not csv_path.exists() or not self._validate_csv_headers(csv_path)

        if csv_path.exists() and not write_header:
            self._cleanup_old_csv_records(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(self.MERGED_BLOCK_INFOS_HEADERS)

            timestamp = datetime.now().isoformat()
            for merged_info in merged_block_infos:
                # warped_mask_shape, warped_mask_contour 처리
                warped_mask_shape = merged_info.get("warped_mask_shape", [])
                warped_mask_contour = merged_info.get("warped_mask_contour", [])

                row = [
                    batch_id,
                    self._serialize_data(merged_info.get("vision_sensor_id", [])),
                    self._serialize_data(merged_info.get("top1_class", [])),
                    self._serialize_data(merged_info.get("bbox_coordinate", [])),
                    self._serialize_data(warped_mask_shape),
                    self._serialize_data(warped_mask_contour),
                    self._serialize_data(merged_info.get("stage_labels", [])),
                    merged_info.get("image_path", ""),
                    self._serialize_data(merged_info.get("image_paths", [])),
                    self._serialize_data(merged_info.get("original_bbox_coordinates", [])),
                    self._serialize_data(merged_info.get("before_projected_bboxes_xywh", [])),
                    self._serialize_data(merged_info.get("matched_text_recognitions", [])),
                    self._serialize_data(merged_info.get("matched_object_counts", {})),
                    self._serialize_data(merged_info.get("matched_ships_counts", {})),
                    self._serialize_data(merged_info.get("matched_blocks_counts", {})),
                    self._serialize_data(merged_info.get("source_infos", [])),
                    merged_info.get("is_merged", 1),
                    self._serialize_data(merged_info.get("merged_block_idx", [])),
                    merged_info.get("voted_ship", ""),
                    merged_info.get("voted_block", ""),
                    merged_info.get("block_idx_in_layout", ""),
                    merged_info.get("merge_group_id", ""),
                    timestamp,
                ]
                writer.writerow(row)

        logger.info(f"[{bay_id}] CSV 저장 완료: {csv_path}")
        return csv_path

    def load_results(self, result_path: str) -> Optional[Dict[str, Any]]:
        """저장된 Pickle 결과 로드"""
        try:
            path = Path(result_path)
            if not path.exists():
                logger.error(f"파일 없음: {result_path}")
                return None

            try:
                with gzip.open(path, "rb") as f:
                    data = pickle.load(f)
            except gzip.BadGzipFile:
                with open(path, "rb") as f:
                    data = pickle.load(f)

            return data

        except Exception as e:
            logger.error(f"결과 로드 실패: {e}")
            return None

    def get_result_path(self, batch_id: str, bay_id: str) -> str:
        """결과 파일 경로 생성"""
        return str(self.base_path / "fusion" / bay_id / batch_id / "result.pkl")
