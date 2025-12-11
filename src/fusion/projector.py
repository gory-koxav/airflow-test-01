# -*- coding: utf-8 -*-
"""
Projector

카메라 좌표계에서 공장 좌표계(Bird's-eye view)로 투영 변환을 수행합니다.
"""

import numpy as np
import cv2
from typing import List, Optional, Set, Dict, Any
import logging

from .models import FrameData, ProjectedData

logger = logging.getLogger(__name__)


class Projector:
    """
    단일 카메라의 프레임 데이터를 절대 좌표계로 투영(사영 변환)하는 클래스
    """

    def __init__(
        self,
        camera_intrinsics: Dict[str, float],
        camera_extrinsics: Dict[str, Dict[str, Any]],
        pixels_per_meter: int = 10,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            camera_intrinsics: 카메라 내부 파라미터 (fx, fy, cx, cy)
            camera_extrinsics: 카메라별 외부 파라미터 (pan, tilt, coord)
            pixels_per_meter: 월드 좌표계 1미터를 몇 픽셀로 표현할지
            target_classes: 투영할 객체의 클래스 목록 (None이면 모든 객체)
        """
        self.fx = camera_intrinsics.get("fx", 1000.0)
        self.fy = camera_intrinsics.get("fy", 1000.0)
        self.cx = camera_intrinsics.get("cx", 960.0)
        self.cy = camera_intrinsics.get("cy", 540.0)
        self.camera_extrinsics = camera_extrinsics
        self.pixels_per_meter = pixels_per_meter
        self.target_classes: Optional[Set[str]] = (
            set(target_classes) if target_classes else None
        )

    def _create_empty_projected_data(
        self, frame_data: FrameData, is_valid: bool = False
    ) -> ProjectedData:
        """빈 ProjectedData 객체 생성"""
        return ProjectedData(
            image_id=frame_data.image_id,
            camera_name=frame_data.camera_name,
            image_path=frame_data.image_path,
            captured_at=frame_data.captured_at,
            warped_image=None,
            warped_masks=[],
            warped_pinjig_masks=[],
            projected_boxes=[],
            projected_boxes_labels=[],
            before_projected_bboxes_xywh=[],
            projected_assembly_boxes=[],
            projected_assembly_labels=[],
            extent=[0, 0, 0, 0],
            clip_polygon=np.array([]),
            is_valid=is_valid,
            merged_boxes=[],
        )

    def _rotation_matrix(self, pan_deg: float, tilt_deg: float) -> np.ndarray:
        """Pan(Yaw), Tilt(Pitch) 각도로 회전 행렬 생성"""
        pan = np.deg2rad(pan_deg)
        tilt = np.deg2rad(tilt_deg)
        R_tilt = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt), -np.sin(tilt)],
                [0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        R_pan = np.array(
            [
                [np.cos(pan), -np.sin(pan), 0],
                [np.sin(pan), np.cos(pan), 0],
                [0, 0, 1],
            ]
        )
        return R_pan @ R_tilt

    def _project_pixel_to_ground(
        self, u: float, v: float, R: np.ndarray, t: np.ndarray
    ) -> Optional[np.ndarray]:
        """픽셀 좌표 (u, v)를 지면(z=0) 좌표로 투영"""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        ray_cam = np.array([x, y, -1.0])
        ray_world = R @ ray_cam

        if np.abs(ray_world[2]) < 1e-6:
            return None
        lam = -t[2] / ray_world[2]
        if lam < 0:
            return None

        intersect = t + lam * ray_world
        return intersect[:2]

    def project(self, frame_data: FrameData, image: Optional[np.ndarray] = None) -> ProjectedData:
        """
        FrameData를 평면도(Bird's-eye view)로 투영

        Args:
            frame_data: 입력 프레임 데이터
            image: 이미지 데이터 (없으면 image_path에서 로드)

        Returns:
            ProjectedData: 투영된 결과
        """
        cam_config = self.camera_extrinsics.get(frame_data.camera_name)
        if not cam_config:
            logger.warning(
                f"[{frame_data.camera_name}] 카메라 설정을 찾을 수 없습니다."
            )
            return self._create_empty_projected_data(frame_data, is_valid=False)

        # 이미지 로드
        if image is None:
            img = cv2.imread(frame_data.image_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(
                    f"[{frame_data.camera_name}] 이미지 로드 실패: {frame_data.image_path}"
                )
                return self._create_empty_projected_data(frame_data, is_valid=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image

        h_img, w_img = img.shape[:2]

        R = self._rotation_matrix(cam_config["pan"], cam_config["tilt"])
        t = np.array(cam_config["coord"])

        # 이미지 네 꼭짓점을 월드 좌표로 투영
        src_corners = np.array(
            [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32
        )
        projected_corners = [
            self._project_pixel_to_ground(u, v, R, t) for u, v in src_corners
        ]

        if any(corner is None for corner in projected_corners):
            logger.warning(
                f"[{frame_data.camera_name}] 일부 영역이 지평선 너머로 투영됨"
            )
            return self._create_empty_projected_data(frame_data, is_valid=False)

        dst_corners = np.array(projected_corners, dtype=np.float32)

        # Homography 계산
        H_mat, _ = cv2.findHomography(src_corners, dst_corners)
        if H_mat is None:
            logger.warning(
                f"[{frame_data.camera_name}] Homography 행렬 계산 실패"
            )
            return self._create_empty_projected_data(frame_data, is_valid=False)

        min_x, max_x = np.min(dst_corners[:, 0]), np.max(dst_corners[:, 0])
        min_y, max_y = np.min(dst_corners[:, 1]), np.max(dst_corners[:, 1])

        warp_width = int(np.ceil((max_x - min_x) * self.pixels_per_meter))
        warp_height = int(np.ceil((max_y - min_y) * self.pixels_per_meter))

        if not (0 < warp_width < 8000 and 0 < warp_height < 8000):
            logger.warning(
                f"[{frame_data.camera_name}] 변환 결과 크기가 너무 큼: {warp_width}x{warp_height}"
            )
            return self._create_empty_projected_data(frame_data, is_valid=False)

        T_matrix = np.array(
            [
                [self.pixels_per_meter, 0, -min_x * self.pixels_per_meter],
                [0, self.pixels_per_meter, -min_y * self.pixels_per_meter],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        H_warp = T_matrix @ H_mat

        # 이미지 Warp
        img_flipped = cv2.flip(cv2.flip(img, 0), 1)
        warped_image = cv2.warpPerspective(
            img_flipped, H_warp, (warp_width, warp_height), flags=cv2.INTER_LINEAR
        )

        # Boundary 마스크 워핑
        warped_masks = []
        for mask_info in frame_data.boundary_masks:
            mask = mask_info.get("boundary_mask")
            if mask is not None:
                mask_flipped = cv2.flip(cv2.flip(mask, 0), 1)
                warped_mask = cv2.warpPerspective(
                    mask_flipped,
                    H_warp,
                    (warp_width, warp_height),
                    flags=cv2.INTER_NEAREST,
                )
                warped_masks.append(warped_mask)

        # Pinjig 마스크 워핑
        warped_pinjig_masks = []
        for mask in frame_data.pinjig_masks:
            mask_flipped = cv2.flip(cv2.flip(mask, 0), 1)
            warped_pinjig_mask = cv2.warpPerspective(
                mask_flipped,
                H_warp,
                (warp_width, warp_height),
                flags=cv2.INTER_NEAREST,
            )
            warped_pinjig_masks.append(warped_pinjig_mask)

        # 객체 경계 상자 투영
        projected_boxes = []
        projected_boxes_labels = []
        before_projected_bboxes_xywh = []

        for det in frame_data.detections:
            class_name = det.get("class_name")
            if self.target_classes and class_name not in self.target_classes:
                continue

            bbox_xywh = det.get("bbox_xywh", [])
            if len(bbox_xywh) != 4:
                continue

            x_min, y_min, w_box, h_box = bbox_xywh
            before_projected_bboxes_xywh.append(bbox_xywh)
            x_max, y_max = x_min + w_box, y_min + h_box

            box_corners_orig = np.array(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.float32,
            )

            # 플립 적용
            box_corners_flipped = []
            for u, v in box_corners_orig:
                flipped_u = w_img - 1 - u
                flipped_v = h_img - 1 - v
                box_corners_flipped.append([flipped_u, flipped_v])

            box_corners_flipped = np.array(box_corners_flipped, dtype=np.float32)

            # Homography 변환
            box_corners_transformed = cv2.perspectiveTransform(
                box_corners_flipped.reshape(-1, 1, 2), H_warp
            ).reshape(-1, 2)

            # 월드 좌표로 변환
            box_corners_world = box_corners_transformed / self.pixels_per_meter
            box_corners_world[:, 0] += min_x
            box_corners_world[:, 1] += min_y

            projected_boxes.append(box_corners_world)
            projected_boxes_labels.append(class_name if class_name else "unknown")

        # Assembly classifications 처리
        projected_assembly_boxes = []
        projected_assembly_labels = []

        for assembly_class in frame_data.assembly_classifications:
            detected_bbox_idx = assembly_class.get("detected_bbox_idx")
            if detected_bbox_idx is None:
                continue

            matching_detection = None
            for detection in frame_data.detections:
                if detection.get("detected_bbox_idx") == detected_bbox_idx:
                    matching_detection = detection
                    break

            if not matching_detection:
                continue

            bbox_xywh = matching_detection.get("bbox_xywh", [])
            if len(bbox_xywh) != 4:
                continue

            x_min, y_min, width, height = bbox_xywh
            x_max = x_min + width
            y_max = y_min + height

            box_corners_orig = np.array(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.float32,
            )

            box_corners_flipped = []
            for u, v in box_corners_orig:
                flipped_u = w_img - 1 - u
                flipped_v = h_img - 1 - v
                box_corners_flipped.append([flipped_u, flipped_v])

            box_corners_flipped = np.array(box_corners_flipped, dtype=np.float32)

            box_corners_transformed = cv2.perspectiveTransform(
                box_corners_flipped.reshape(-1, 1, 2), H_warp
            ).reshape(-1, 2)

            box_corners_world = box_corners_transformed / self.pixels_per_meter
            box_corners_world[:, 0] += min_x
            box_corners_world[:, 1] += min_y

            projected_assembly_boxes.append(box_corners_world)

            # 중심점 계산 및 변환
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            flipped_center_x = w_img - 1 - center_x
            flipped_center_y = h_img - 1 - center_y

            center_point = np.array(
                [[[flipped_center_x, flipped_center_y]]], dtype=np.float32
            )
            projected_center = cv2.perspectiveTransform(center_point, H_warp)[0, 0]

            world_center_x = projected_center[0] / self.pixels_per_meter + min_x
            world_center_y = projected_center[1] / self.pixels_per_meter + min_y

            projected_assembly_labels.append(
                {
                    "position": [world_center_x, world_center_y],
                    "label": assembly_class.get("top1_class", "Unknown"),
                    "confidence": assembly_class.get("top1_confidence", 0.0),
                    "detected_bbox_idx": detected_bbox_idx,
                }
            )

        return ProjectedData(
            image_id=frame_data.image_id,
            camera_name=frame_data.camera_name,
            image_path=frame_data.image_path,
            captured_at=frame_data.captured_at,
            is_valid=True,
            warped_image=warped_image,
            warped_masks=warped_masks,
            warped_pinjig_masks=warped_pinjig_masks,
            projected_boxes=projected_boxes,
            projected_boxes_labels=projected_boxes_labels,
            before_projected_bboxes_xywh=before_projected_bboxes_xywh,
            projected_assembly_boxes=projected_assembly_boxes,
            projected_assembly_labels=projected_assembly_labels,
            extent=[min_x, max_x, max_y, min_y],
            clip_polygon=dst_corners,
            merged_boxes=[],
        )
