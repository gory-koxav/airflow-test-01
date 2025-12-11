# -*- coding: utf-8 -*-
"""
Result Saver

Observation 결과를 Pickle 파일로 저장합니다.
Observer와 Processor 간 데이터 공유를 위해 공유 스토리지에 저장합니다.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pickle
import logging
import os
import gzip
from datetime import datetime

from .image_provider.base import CapturedImage
from .inference_service import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class SaveResult:
    """저장 결과 데이터 모델"""
    success: bool
    result_path: str
    batch_id: str
    bay_id: str
    processed_count: int
    error_message: Optional[str] = None


class ResultSaver:
    """
    Observation 결과를 Pickle 파일로 저장합니다.

    저장 경로 구조:
    - {base_path}/observation/{bay_id}/{batch_id}/result.pkl
    """

    def __init__(
        self,
        base_path: str = "/opt/airflow/data",
        compress: bool = True,
        compression_level: int = 6,
    ):
        """
        Args:
            base_path: 공유 스토리지 기본 경로
            compress: 압축 사용 여부
            compression_level: gzip 압축 레벨 (1-9)
        """
        self.base_path = Path(base_path)
        self.compress = compress
        self.compression_level = compression_level

    def save_results(
        self,
        captured_images: List[CapturedImage],
        inference_results: Dict[str, InferenceResult],
        batch_id: str,
        bay_id: str,
    ) -> SaveResult:
        """
        캡처된 이미지와 추론 결과를 Pickle 파일로 저장합니다.

        Args:
            captured_images: 캡처된 이미지 리스트
            inference_results: 추론 결과 딕셔너리 (image_id -> InferenceResult)
            batch_id: 배치 식별자
            bay_id: Bay 식별자

        Returns:
            SaveResult: 저장 결과
        """
        try:
            # 저장 디렉토리 생성
            save_dir = self.base_path / "observation" / bay_id / batch_id
            save_dir.mkdir(parents=True, exist_ok=True)

            # 저장할 데이터 구성
            save_data = self._prepare_save_data(
                captured_images, inference_results, batch_id, bay_id
            )

            # Pickle 파일 저장
            result_path = save_dir / "result.pkl"
            self._save_pickle(save_data, result_path)

            logger.info(f"[{bay_id}] Results saved to {result_path}")

            return SaveResult(
                success=True,
                result_path=str(result_path),
                batch_id=batch_id,
                bay_id=bay_id,
                processed_count=len(captured_images),
            )

        except Exception as e:
            logger.error(f"[{bay_id}] Failed to save results: {e}")
            return SaveResult(
                success=False,
                result_path="",
                batch_id=batch_id,
                bay_id=bay_id,
                processed_count=0,
                error_message=str(e),
            )

    def _prepare_save_data(
        self,
        captured_images: List[CapturedImage],
        inference_results: Dict[str, InferenceResult],
        batch_id: str,
        bay_id: str,
    ) -> Dict[str, Any]:
        """저장할 데이터 준비"""
        # 이미지 데이터를 딕셔너리로 변환
        images_data = {}
        for img in captured_images:
            images_data[img.image_id] = {
                "camera_name": img.camera_name,
                "bay_id": img.bay_id,
                "image_data": img.image_data,  # numpy array
                "captured_at": img.captured_at.isoformat(),
                "metadata": img.metadata,
            }

        # 추론 결과를 딕셔너리로 변환
        results_data = {}
        for image_id, result in inference_results.items():
            results_data[image_id] = {
                "camera_name": result.camera_name,
                "detections": result.detections,
                "boundary_masks": self._serialize_masks(result.boundary_masks),
                "assembly_classifications": result.assembly_classifications,
                "pinjig_masks": [m.tolist() if isinstance(m, np.ndarray) else m for m in result.pinjig_masks],
                "pinjig_classifications": result.pinjig_classifications,
                "metadata": result.metadata,
            }

        return {
            "batch_id": batch_id,
            "bay_id": bay_id,
            "saved_at": datetime.now().isoformat(),
            "images": images_data,
            "inference_results": results_data,
            "summary": {
                "total_images": len(captured_images),
                "total_detections": sum(
                    len(r.detections) for r in inference_results.values()
                ),
                "camera_names": [img.camera_name for img in captured_images],
            },
        }

    def _serialize_masks(
        self,
        masks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """마스크 데이터 직렬화"""
        serialized = []
        for mask_dict in masks:
            serialized_mask = {}
            for key, value in mask_dict.items():
                if isinstance(value, np.ndarray):
                    serialized_mask[key] = value.tolist()
                else:
                    serialized_mask[key] = value
            serialized.append(serialized_mask)
        return serialized

    def _save_pickle(
        self,
        data: Dict[str, Any],
        file_path: Path,
    ) -> None:
        """Pickle 파일 저장"""
        if self.compress:
            with gzip.open(file_path, 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_results(
        self,
        result_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        저장된 결과를 로드합니다.

        Args:
            result_path: Pickle 파일 경로

        Returns:
            Optional[Dict[str, Any]]: 로드된 데이터 또는 None
        """
        try:
            path = Path(result_path)
            if not path.exists():
                logger.error(f"Result file not found: {result_path}")
                return None

            # 압축 여부 자동 감지
            try:
                with gzip.open(path, 'rb') as f:
                    data = pickle.load(f)
            except gzip.BadGzipFile:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

            return data

        except Exception as e:
            logger.error(f"Failed to load results from {result_path}: {e}")
            return None

    def get_result_path(
        self,
        batch_id: str,
        bay_id: str,
    ) -> str:
        """결과 파일 경로 생성"""
        return str(self.base_path / "observation" / bay_id / batch_id / "result.pkl")

    def cleanup_old_results(
        self,
        bay_id: str,
        max_files: int = 100,
    ) -> int:
        """
        오래된 결과 파일 정리

        Args:
            bay_id: Bay 식별자
            max_files: 최대 유지할 파일 수

        Returns:
            int: 삭제된 파일 수
        """
        bay_dir = self.base_path / "observation" / bay_id
        if not bay_dir.exists():
            return 0

        # batch_id 디렉토리 목록 (수정 시간 기준 정렬)
        batch_dirs = sorted(
            [d for d in bay_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        deleted_count = 0
        for batch_dir in batch_dirs[max_files:]:
            try:
                # 디렉토리 내 모든 파일 삭제
                for file in batch_dir.iterdir():
                    file.unlink()
                batch_dir.rmdir()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {batch_dir}: {e}")

        if deleted_count > 0:
            logger.info(f"[{bay_id}] Cleaned up {deleted_count} old result directories")

        return deleted_count


# numpy import for serialization
import numpy as np
