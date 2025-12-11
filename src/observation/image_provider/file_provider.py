# -*- coding: utf-8 -*-
"""
File Image Provider

저장된 이미지 파일에서 이미지를 로드합니다.
백테스트 및 개발/테스트 용도로 사용됩니다.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re
import cv2
import numpy as np

from .base import ImageProvider, CapturedImage

logger = logging.getLogger(__name__)


class FileImageProvider(ImageProvider):
    """
    저장된 이미지 파일에서 이미지를 로드합니다.
    백테스트 및 개발/테스트 용도로 사용됩니다.
    """

    # 타임스탬프 패턴: YYYYMMDDHHMMSS
    TIMESTAMP_PATTERN = re.compile(r'(\d{14})')

    # 대체 타임스탬프 패턴: YYYY-MM-DD_HH-MM-SS 또는 YYYYMMDD-HHMMSS
    ALT_TIMESTAMP_PATTERNS = [
        re.compile(r'(\d{8})-(\d{6})'),  # YYYYMMDD-HHMMSS
        re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})'),  # YYYY-MM-DD_HH-MM-SS
    ]

    def __init__(
        self,
        base_path: str,
        time_tolerance_hours: float = 1.0,
        file_extensions: Optional[List[str]] = None,
    ):
        """
        Args:
            base_path: 이미지 파일이 저장된 기본 디렉토리 경로
            time_tolerance_hours: 시간 허용 오차 (시간 단위)
            file_extensions: 허용할 파일 확장자 목록 (기본: ['.png', '.jpg', '.jpeg'])
        """
        self.base_path = Path(base_path)
        self.time_tolerance = timedelta(hours=time_tolerance_hours)
        self.file_extensions = file_extensions or ['.png', '.jpg', '.jpeg']

        # 초기화 시 base_path 존재 여부 로깅
        logger.info(f"FileImageProvider initialized with base_path: {self.base_path}")
        if not self.base_path.exists():
            logger.warning(f"Base path does not exist: {self.base_path}")
        else:
            logger.info(f"Base path exists: {self.base_path}")

    def get_images(
        self,
        reference_time: datetime,
        cameras: List[Dict[str, Any]],
        bay_id: str,
    ) -> List[CapturedImage]:
        """기준 시간에 맞는 저장된 이미지 파일을 로드합니다."""
        logger.info(f"[{bay_id}] get_images called - reference_time: {reference_time}, cameras: {len(cameras) if cameras else 0}")

        # base_path 존재 여부 체크
        if not self.base_path.exists():
            logger.error(
                f"[{bay_id}] CRITICAL: Base path does not exist: {self.base_path}. "
                f"Please ensure the NFS mount or image directory is available."
            )
            return []

        if not cameras:
            logger.warning(f"[{bay_id}] No cameras configured")
            return []

        logger.info(f"[{bay_id}] Configured cameras: {[c.get('name', 'unknown') for c in cameras]}")

        captured_images: List[CapturedImage] = []
        failed_cameras: List[str] = []

        for camera_config in cameras:
            camera_name = camera_config.get("name", "unknown")
            try:
                image = self._load_image_for_camera(
                    camera_name, reference_time, bay_id
                )
                if image:
                    captured_images.append(image)
                    logger.info(f"[{bay_id}] Successfully loaded image for camera: {camera_name}")
                else:
                    failed_cameras.append(camera_name)
                    logger.warning(
                        f"[{bay_id}] No matching image found for camera: {camera_name}"
                    )
            except Exception as e:
                failed_cameras.append(camera_name)
                logger.error(
                    f"[{bay_id}] Camera {camera_name} file load failed: {e}",
                    exc_info=True
                )

        # 최종 결과 요약 로깅
        logger.info(
            f"[{bay_id}] Image loading summary: "
            f"success={len(captured_images)}/{len(cameras)}, "
            f"failed={len(failed_cameras)}"
        )

        if failed_cameras:
            logger.warning(
                f"[{bay_id}] Failed cameras: {failed_cameras}"
            )

        return captured_images

    def _load_image_for_camera(
        self,
        camera_name: str,
        reference_time: datetime,
        bay_id: str,
    ) -> Optional[CapturedImage]:
        """특정 카메라의 기준 시간에 가장 가까운 이미지를 로드합니다."""
        # 카메라별 디렉토리 탐색 (여러 경로 시도)
        search_paths = [
            self.base_path / camera_name,                    # {base_path}/{camera_name}
            self.base_path / bay_id / camera_name,           # {base_path}/{bay_id}/{camera_name}
            self.base_path / bay_id,                         # {base_path}/{bay_id} (파일명에 카메라 이름 포함)
        ]

        logger.debug(f"[{bay_id}] Searching for camera '{camera_name}' images in paths: {search_paths}")

        camera_dir = None
        for path in search_paths:
            if path.exists() and path.is_dir():
                camera_dir = path
                logger.debug(f"[{bay_id}] Found directory for camera '{camera_name}': {path}")
                break
            else:
                logger.debug(f"[{bay_id}] Path does not exist: {path}")

        if camera_dir is None:
            # 대안: 전체 base_path에서 카메라 이름이 포함된 파일 검색
            logger.info(
                f"[{bay_id}] No dedicated directory found for camera '{camera_name}'. "
                f"Searching in base_path: {self.base_path}"
            )
            return self._search_in_base_path(camera_name, reference_time, bay_id)

        # 이미지 파일 목록 수집
        image_files = self._collect_image_files(camera_dir)
        logger.debug(f"[{bay_id}] Found {len(image_files)} image files in {camera_dir}")

        if not image_files:
            logger.warning(
                f"[{bay_id}] No image files found for camera '{camera_name}' in directory: {camera_dir}"
            )
            # 디렉토리 내용 로깅 (디버깅용)
            try:
                contents = list(camera_dir.iterdir())[:10]  # 최대 10개만
                logger.debug(f"[{bay_id}] Directory contents (first 10): {[f.name for f in contents]}")
            except Exception:
                pass
            return None

        # 가장 가까운 이미지 찾기
        best_file = self._find_closest_image(image_files, reference_time)
        if best_file is None:
            logger.warning(
                f"[{bay_id}] No image file within time tolerance for camera '{camera_name}'. "
                f"Reference time: {reference_time}, Tolerance: {self.time_tolerance}"
            )
            return None

        logger.debug(f"[{bay_id}] Selected best matching file: {best_file}")
        return self._load_image_file(best_file, camera_name, bay_id, reference_time)

    def _search_in_base_path(
        self,
        camera_name: str,
        reference_time: datetime,
        bay_id: str,
    ) -> Optional[CapturedImage]:
        """base_path 전체에서 카메라 이름이 포함된 파일 검색"""
        logger.debug(f"[{bay_id}] Searching entire base_path for files containing '{camera_name}'")

        matching_files = []

        for ext in self.file_extensions:
            # 카메라 이름이 포함된 파일 검색
            pattern = f"*{camera_name}*{ext}"
            found = list(self.base_path.rglob(pattern))
            matching_files.extend(found)
            logger.debug(f"[{bay_id}] Pattern '{pattern}': found {len(found)} files")

        if not matching_files:
            logger.warning(
                f"[{bay_id}] No files found containing camera name '{camera_name}' "
                f"in base_path: {self.base_path}"
            )
            # base_path 디렉토리 구조 로깅 (디버깅용)
            try:
                subdirs = [d.name for d in self.base_path.iterdir() if d.is_dir()][:10]
                logger.info(f"[{bay_id}] Available subdirectories in base_path (first 10): {subdirs}")
            except Exception as e:
                logger.debug(f"[{bay_id}] Could not list base_path contents: {e}")
            return None

        logger.debug(f"[{bay_id}] Found {len(matching_files)} total files for camera '{camera_name}'")

        best_file = self._find_closest_image(matching_files, reference_time)
        if best_file is None:
            logger.warning(
                f"[{bay_id}] No matching file within time tolerance for camera '{camera_name}'"
            )
            return None

        return self._load_image_file(best_file, camera_name, bay_id, reference_time)

    def _collect_image_files(self, directory: Path) -> List[Path]:
        """디렉토리에서 이미지 파일 목록 수집"""
        image_files = []
        for ext in self.file_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        return image_files

    def _find_closest_image(
        self,
        files: List[Path],
        reference_time: datetime,
    ) -> Optional[Path]:
        """기준 시간에 가장 가까운 이미지 파일 찾기"""
        best_file = None
        best_diff = self.time_tolerance

        for file_path in files:
            timestamp = self._extract_timestamp(file_path.name)
            if timestamp is None:
                # 타임스탬프가 없는 파일도 후보로 고려
                if best_file is None:
                    best_file = file_path
                continue

            diff = abs(timestamp - reference_time)
            if diff <= self.time_tolerance and diff < best_diff:
                best_diff = diff
                best_file = file_path

        return best_file

    def _extract_timestamp(self, filename: str) -> Optional[datetime]:
        """파일 이름에서 타임스탬프 추출"""
        # 기본 패턴: YYYYMMDDHHMMSS
        match = self.TIMESTAMP_PATTERN.search(filename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
            except ValueError:
                pass

        # 대체 패턴: YYYYMMDD-HHMMSS
        match = self.ALT_TIMESTAMP_PATTERNS[0].search(filename)
        if match:
            try:
                date_str = match.group(1)
                time_str = match.group(2)
                return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            except ValueError:
                pass

        # 대체 패턴: YYYY-MM-DD_HH-MM-SS
        match = self.ALT_TIMESTAMP_PATTERNS[1].search(filename)
        if match:
            try:
                return datetime(
                    int(match.group(1)),  # year
                    int(match.group(2)),  # month
                    int(match.group(3)),  # day
                    int(match.group(4)),  # hour
                    int(match.group(5)),  # minute
                    int(match.group(6)),  # second
                )
            except ValueError:
                pass

        return None

    def _load_image_file(
        self,
        file_path: Path,
        camera_name: str,
        bay_id: str,
        reference_time: datetime,
    ) -> Optional[CapturedImage]:
        """이미지 파일 로드"""
        image_data = cv2.imread(str(file_path))
        if image_data is None:
            logger.error(f"Failed to load image: {file_path}")
            return None

        # 파일에서 타임스탬프 추출 또는 기준 시간 사용
        file_timestamp = self._extract_timestamp(file_path.name)
        captured_at = file_timestamp or reference_time

        image_id = f"{bay_id}_{camera_name}_{captured_at.strftime('%Y%m%d%H%M%S')}"

        return CapturedImage(
            image_id=image_id,
            camera_name=camera_name,
            bay_id=bay_id,
            image_data=image_data,
            captured_at=captured_at,
            metadata={
                "source_file": str(file_path),
                "capture_method": "file",
            },
        )

    def validate_config(self, cameras: List[Dict[str, Any]]) -> bool:
        """카메라 설정 유효성 검사"""
        if not cameras:
            return False

        for camera in cameras:
            if "name" not in camera or not camera["name"]:
                return False

        return True

    def set_base_path(self, new_path: str) -> None:
        """기본 경로 변경"""
        self.base_path = Path(new_path)

    def list_available_timestamps(
        self,
        camera_name: str,
    ) -> List[datetime]:
        """특정 카메라의 사용 가능한 타임스탬프 목록 반환"""
        camera_dir = self.base_path / camera_name
        if not camera_dir.exists():
            return []

        timestamps = []
        image_files = self._collect_image_files(camera_dir)

        for file_path in image_files:
            timestamp = self._extract_timestamp(file_path.name)
            if timestamp:
                timestamps.append(timestamp)

        return sorted(timestamps)
