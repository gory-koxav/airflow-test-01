# -*- coding: utf-8 -*-
"""
Overlap Analyzer

서로 다른 카메라에서 투영된 객체들의 중첩을 분석하고 병합합니다.
"""

import numpy as np
import uuid
from typing import List, Dict, Tuple, Any
from itertools import combinations
import logging

from .models import ProjectedData

logger = logging.getLogger(__name__)

# Shapely 가용성 확인
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely 라이브러리가 없습니다. 대체 알고리즘을 사용합니다.")


class OverlapAnalyzer:
    """
    서로 다른 카메라의 ProjectedData에서 객체 중첩을 분석하고 병합하는 클래스
    """

    def __init__(self, iou_threshold: float = 0.3):
        """
        Args:
            iou_threshold: IOU가 이 값 이상일 때 박스들을 병합
        """
        self.iou_threshold = iou_threshold
        logger.info(f"OverlapAnalyzer 초기화 (IOU Threshold: {iou_threshold})")

    def _calculate_polygon_iou_shapely(
        self, poly1: np.ndarray, poly2: np.ndarray
    ) -> float:
        """Shapely를 사용한 다각형 IOU 계산"""
        try:
            polygon1 = ShapelyPolygon(poly1)
            polygon2 = ShapelyPolygon(poly2)

            if not polygon1.is_valid:
                polygon1 = make_valid(polygon1)
            if not polygon2.is_valid:
                polygon2 = make_valid(polygon2)

            intersection = polygon1.intersection(polygon2)
            union = polygon1.union(polygon2)

            if union.area == 0:
                return 0.0

            return intersection.area / union.area

        except Exception as e:
            logger.warning(f"Shapely IOU 계산 오류: {e}")
            return 0.0

    def _calculate_polygon_iou_fallback(
        self, poly1: np.ndarray, poly2: np.ndarray
    ) -> float:
        """Shapely가 없을 때 사용하는 대체 IOU 계산"""

        def polygon_area(vertices):
            """Shoelace formula를 사용한 다각형 면적 계산"""
            n = len(vertices)
            if n < 3:
                return 0.0
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i][0] * vertices[j][1]
                area -= vertices[j][0] * vertices[i][1]
            return abs(area) / 2.0

        def line_intersection(p1, p2, p3, p4):
            """두 선분의 교점 계산"""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

            if 0 <= t <= 1 and 0 <= u <= 1:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return [x, y]
            return None

        def sutherland_hodgman_clip(subject_polygon, clip_polygon):
            """Sutherland-Hodgman 알고리즘으로 다각형 교집합 구하기"""

            def inside_edge(point, edge_start, edge_end):
                return (
                    (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
                    - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
                ) >= 0

            output_list = list(subject_polygon)

            for i in range(len(clip_polygon)):
                if len(output_list) == 0:
                    break

                input_list = output_list
                output_list = []

                edge_start = clip_polygon[i]
                edge_end = clip_polygon[(i + 1) % len(clip_polygon)]

                for j in range(len(input_list)):
                    current_vertex = input_list[j]
                    previous_vertex = input_list[j - 1]

                    if inside_edge(current_vertex, edge_start, edge_end):
                        if not inside_edge(previous_vertex, edge_start, edge_end):
                            intersection = line_intersection(
                                previous_vertex, current_vertex, edge_start, edge_end
                            )
                            if intersection:
                                output_list.append(intersection)
                        output_list.append(current_vertex)
                    elif inside_edge(previous_vertex, edge_start, edge_end):
                        intersection = line_intersection(
                            previous_vertex, current_vertex, edge_start, edge_end
                        )
                        if intersection:
                            output_list.append(intersection)

            return output_list

        intersection_vertices = sutherland_hodgman_clip(poly1.tolist(), poly2.tolist())

        if len(intersection_vertices) < 3:
            return 0.0

        area1 = polygon_area(poly1)
        area2 = polygon_area(poly2)
        intersection_area = polygon_area(intersection_vertices)

        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """두 경계 상자 간의 IOU 계산"""
        if SHAPELY_AVAILABLE:
            return self._calculate_polygon_iou_shapely(box1, box2)
        else:
            return self._calculate_polygon_iou_fallback(box1, box2)

    def _create_enclosing_box(self, boxes: List[np.ndarray]) -> np.ndarray:
        """여러 박스를 포함하는 최소 경계 상자 생성"""
        all_points = np.vstack(boxes)

        min_x = np.min(all_points[:, 0])
        max_x = np.max(all_points[:, 0])
        min_y = np.min(all_points[:, 1])
        max_y = np.max(all_points[:, 1])

        enclosing_box = np.array(
            [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        )

        return enclosing_box

    def analyze_overlaps(
        self, projected_data_list: List[ProjectedData]
    ) -> List[ProjectedData]:
        """
        ProjectedData 리스트를 분석하여 겹치는 박스들을 찾고 병합

        Args:
            projected_data_list: 분석할 ProjectedData 리스트

        Returns:
            병합된 박스 정보가 추가된 ProjectedData 리스트
        """
        logger.info("Overlap Analysis 시작")

        valid_data_list = [data for data in projected_data_list if data.is_valid]

        if len(valid_data_list) < 2:
            logger.warning(
                "유효한 카메라 데이터가 2개 미만이므로 overlap 분석을 건너뜁니다."
            )
            return projected_data_list

        logger.info(f"총 {len(valid_data_list)}개의 유효한 카메라 데이터를 분석합니다.")

        merged_boxes = []
        merge_mapping = {}  # (camera_name, box_idx) -> merge_group_id

        # 모든 카메라 쌍에 대해 비교
        for i, j in combinations(range(len(valid_data_list)), 2):
            data1 = valid_data_list[i]
            data2 = valid_data_list[j]

            logger.debug(
                f"카메라 '{data1.camera_name}' vs '{data2.camera_name}' 비교"
            )

            if len(data1.projected_boxes) == 0 or len(data2.projected_boxes) == 0:
                continue

            overlapping_groups = []

            for idx1, box1 in enumerate(data1.projected_boxes):
                for idx2, box2 in enumerate(data2.projected_boxes):
                    iou = self._calculate_iou(box1, box2)

                    if iou >= self.iou_threshold:
                        logger.debug(
                            f"IOU >= threshold: [{data1.camera_name}:{idx1}] <-> [{data2.camera_name}:{idx2}] = {iou:.4f}"
                        )

                        # 기존 그룹에 추가할지 새 그룹을 만들지 결정
                        added_to_group = False
                        for group in overlapping_groups:
                            if any(
                                np.array_equal(box, box1) or np.array_equal(box, box2)
                                for box in group["boxes"]
                            ):
                                if not any(
                                    np.array_equal(box, box1) for box in group["boxes"]
                                ):
                                    group["boxes"].append(box1)
                                    group["sources"].append(
                                        (data1.camera_name, idx1)
                                    )
                                if not any(
                                    np.array_equal(box, box2) for box in group["boxes"]
                                ):
                                    group["boxes"].append(box2)
                                    group["sources"].append(
                                        (data2.camera_name, idx2)
                                    )
                                added_to_group = True
                                break

                        if not added_to_group:
                            group_id = f"merge_{uuid.uuid4().hex[:8]}"
                            overlapping_groups.append(
                                {
                                    "id": group_id,
                                    "boxes": [box1, box2],
                                    "sources": [
                                        (data1.camera_name, idx1),
                                        (data2.camera_name, idx2),
                                    ],
                                }
                            )

            # 각 그룹에 대해 병합 박스 생성
            for group in overlapping_groups:
                merged_box = self._create_enclosing_box(group["boxes"])
                merged_boxes.append(merged_box)

                # 매핑 정보 저장
                for camera_name, box_idx in group["sources"]:
                    merge_mapping[(camera_name, box_idx)] = group["id"]

                logger.info(
                    f"그룹 {group['id']}: {len(group['boxes'])}개 박스 병합"
                )

        logger.info(f"Overlap Analysis 완료: 총 {len(merged_boxes)}개 병합 박스 생성")

        # 병합 정보를 ProjectedData에 추가
        for data in projected_data_list:
            data.merged_boxes = merged_boxes.copy()

            # 마스크-병합그룹 매핑 설정
            for idx in range(len(data.projected_boxes)):
                key = (data.camera_name, idx)
                if key in merge_mapping:
                    data.mask_merge_mapping[idx] = merge_mapping[key]
                else:
                    # 병합되지 않은 박스는 고유 그룹 ID 부여
                    data.mask_merge_mapping[idx] = f"single_{data.camera_name}_{idx}"

        return projected_data_list
