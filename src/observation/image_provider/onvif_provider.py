# -*- coding: utf-8 -*-
"""
ONVIF Image Provider

ONVIF 프로토콜을 사용하여 카메라에서 실시간으로 이미지를 캡처합니다.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import cv2
import requests
from requests.auth import HTTPDigestAuth
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import ImageProvider, CapturedImage

logger = logging.getLogger(__name__)


class OnvifImageProvider(ImageProvider):
    """
    ONVIF 프로토콜을 사용하여 카메라에서 실시간으로 이미지를 캡처합니다.
    """

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 3,
        parallel: bool = True,
        max_workers: int = 6,
    ):
        """
        Args:
            timeout: 카메라 연결 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            parallel: 병렬 캡처 활성화 여부
            max_workers: 병렬 캡처 시 최대 워커 수
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.parallel = parallel
        self.max_workers = max_workers

    def get_images(
        self,
        reference_time: datetime,
        cameras: List[Dict[str, Any]],
        bay_id: str,
    ) -> List[CapturedImage]:
        """모든 카메라에서 실시간으로 이미지를 캡처합니다."""
        if not cameras:
            logger.warning(f"[{bay_id}] No cameras configured")
            return []

        captured_images: List[CapturedImage] = []

        if self.parallel and len(cameras) > 1:
            captured_images = self._capture_parallel(cameras, reference_time, bay_id)
        else:
            captured_images = self._capture_sequential(cameras, reference_time, bay_id)

        logger.info(
            f"[{bay_id}] Captured {len(captured_images)}/{len(cameras)} images"
        )
        return captured_images

    def _capture_parallel(
        self,
        cameras: List[Dict[str, Any]],
        reference_time: datetime,
        bay_id: str,
    ) -> List[CapturedImage]:
        """병렬로 이미지 캡처"""
        captured_images: List[CapturedImage] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_camera = {
                executor.submit(
                    self._capture_single_camera, camera, reference_time, bay_id
                ): camera
                for camera in cameras
            }

            for future in as_completed(future_to_camera):
                camera = future_to_camera[future]
                try:
                    image = future.result()
                    if image:
                        captured_images.append(image)
                except Exception as e:
                    logger.error(
                        f"[{bay_id}] Camera {camera.get('name', 'unknown')} "
                        f"capture failed: {e}"
                    )

        return captured_images

    def _capture_sequential(
        self,
        cameras: List[Dict[str, Any]],
        reference_time: datetime,
        bay_id: str,
    ) -> List[CapturedImage]:
        """순차적으로 이미지 캡처"""
        captured_images: List[CapturedImage] = []

        for camera in cameras:
            try:
                image = self._capture_single_camera(camera, reference_time, bay_id)
                if image:
                    captured_images.append(image)
            except Exception as e:
                logger.error(
                    f"[{bay_id}] Camera {camera.get('name', 'unknown')} "
                    f"capture failed: {e}"
                )

        return captured_images

    def _capture_single_camera(
        self,
        config: Dict[str, Any],
        reference_time: datetime,
        bay_id: str,
    ) -> Optional[CapturedImage]:
        """단일 카메라에서 이미지를 캡처합니다."""
        camera_name = config.get("name", "unknown")
        image_id = f"{bay_id}_{camera_name}_{reference_time.strftime('%Y%m%d%H%M%S')}"

        # 설정 검증
        if not self._validate_single_camera_config(config):
            logger.warning(f"Invalid camera config for {camera_name}")
            return self._create_failure_image(image_id, camera_name, bay_id, reference_time)

        # 스냅샷 시도
        for attempt in range(1, self.max_retries + 1):
            try:
                image_data = self._get_snapshot(config)
                if image_data is not None:
                    return CapturedImage(
                        image_id=image_id,
                        camera_name=camera_name,
                        bay_id=bay_id,
                        image_data=image_data,
                        captured_at=reference_time,
                        metadata={
                            "host": config.get("host"),
                            "port": config.get("port"),
                            "capture_method": "onvif_snapshot",
                        },
                    )
            except Exception as e:
                logger.warning(
                    f"[{bay_id}] Camera {camera_name} snapshot attempt "
                    f"{attempt}/{self.max_retries} failed: {e}"
                )

        # 모든 시도 실패 시 RTSP 폴백
        try:
            image_data = self._get_rtsp_frame(config)
            if image_data is not None:
                return CapturedImage(
                    image_id=image_id,
                    camera_name=camera_name,
                    bay_id=bay_id,
                    image_data=image_data,
                    captured_at=reference_time,
                    metadata={
                        "host": config.get("host"),
                        "port": config.get("port"),
                        "capture_method": "rtsp_fallback",
                    },
                )
        except Exception as e:
            logger.error(f"[{bay_id}] Camera {camera_name} RTSP fallback failed: {e}")

        return self._create_failure_image(image_id, camera_name, bay_id, reference_time)

    def _get_snapshot(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """ONVIF 스냅샷 URI를 통해 이미지 획득"""
        host = config.get("host")
        port = config.get("port", 80)
        username = config.get("username", "admin")
        password = config.get("password", "")

        # 일반적인 ONVIF 스냅샷 URI 패턴들
        snapshot_uris = [
            f"http://{host}:{port}/onvif-http/snapshot",
            f"http://{host}:{port}/Streaming/channels/1/picture",
            f"http://{host}:{port}/cgi-bin/snapshot.cgi",
            f"http://{host}:{port}/snap.jpg",
        ]

        for uri in snapshot_uris:
            try:
                response = requests.get(
                    uri,
                    auth=HTTPDigestAuth(username, password),
                    timeout=self.timeout,
                    stream=True,
                )

                if response.status_code == 200:
                    image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    if image is not None and image.size > 0:
                        return image
            except requests.RequestException:
                continue

        return None

    def _get_rtsp_frame(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """RTSP 스트림에서 프레임 획득 (폴백)"""
        host = config.get("host")
        username = config.get("username", "admin")
        password = config.get("password", "")

        rtsp_urls = [
            f"rtsp://{username}:{password}@{host}:554/Streaming/Channels/1",
            f"rtsp://{username}:{password}@{host}:554/cam/realmonitor?channel=1&subtype=0",
        ]

        for rtsp_url in rtsp_urls:
            try:
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()

                    if ret and frame is not None:
                        return frame
            except Exception:
                continue

        return None

    def _create_failure_image(
        self,
        image_id: str,
        camera_name: str,
        bay_id: str,
        reference_time: datetime,
    ) -> CapturedImage:
        """캡처 실패 시 빈 이미지 생성"""
        # 1920x1080 검은색 이미지 생성
        failure_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 실패 메시지 표시
        cv2.putText(
            failure_image,
            f"CAPTURE FAILED: {camera_name}",
            (500, 540),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )

        return CapturedImage(
            image_id=image_id,
            camera_name=camera_name,
            bay_id=bay_id,
            image_data=failure_image,
            captured_at=reference_time,
            metadata={
                "capture_method": "failure",
                "error": "All capture methods failed",
            },
        )

    def _validate_single_camera_config(self, config: Dict[str, Any]) -> bool:
        """단일 카메라 설정 유효성 검사"""
        required_fields = ["name", "host"]
        for field in required_fields:
            if field not in config or not config[field]:
                return False
        return True

    def validate_config(self, cameras: List[Dict[str, Any]]) -> bool:
        """카메라 설정 유효성 검사"""
        if not cameras:
            return False

        for camera in cameras:
            if not self._validate_single_camera_config(camera):
                return False

        return True
