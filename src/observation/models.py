# -*- coding: utf-8 -*-
"""
Observation 모듈 데이터 모델

이미지 캡처 및 AI 추론을 위한 데이터 구조를 정의합니다.
Legacy visionflow-legacy-observation의 observation/core/models.py를 기반으로 합니다.
"""
import datetime
import numpy as np

from typing import List, Dict, Any, Literal, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Camera:
    """
    CCTV 카메라의 데이터 구조와 기본 동작을 정의하는 데이터 클래스

    Attributes:
        bay: Bay 식별자 (예: "12bay-west", "64bay")
        name: 카메라 이름 (예: "NS6130Y2509S001R1")
        host: 카메라 IP 주소
        port: 카메라 포트 번호
        username: 카메라 로그인 사용자명
        password: 카메라 로그인 비밀번호 (repr에서 제외)
    """
    bay: str
    name: str
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)


@dataclass
class CapturedImage:
    """
    카메라에서 캡처된 이미지와 관련 메타데이터를 담는 데이터 클래스 (DTO)

    Attributes:
        image_id: 각 이미지를 식별할 고유 ID (예: "12bay-west_NS6130Y2509S001R1_20250702161000")
        camera_name: 이미지를 촬영한 카메라 이름
        bay_id: Bay 식별자 (예: "12bay-west", "64bay")
        image_data: 실제 이미지 데이터 (NumPy 배열, BGR 형식)
        captured_at: 촬영 시각
        source_path: 이미지 파일 경로 (XCom 직렬화용)
        metadata: 추가 메타데이터 (선택적)
    """
    image_id: str
    camera_name: str
    bay_id: str
    image_data: np.ndarray
    captured_at: datetime.datetime
    source_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """이미지 데이터 유효성 검사"""
        if self.image_data is None:
            raise ValueError("image_data cannot be None")
        if not isinstance(self.image_data, np.ndarray):
            raise TypeError("image_data must be a numpy array")

    def __repr__(self) -> str:
        """repr 출력 간소화 (이미지 데이터 제외)"""
        return f"CapturedImage(image_id={self.image_id}, camera_name={self.camera_name}, bay_id={self.bay_id})"

    @property
    def shape(self) -> tuple:
        """이미지 shape 반환"""
        return self.image_data.shape

    @property
    def is_valid(self) -> bool:
        """이미지가 유효한지 확인"""
        return (
            self.image_data is not None
            and len(self.image_data.shape) >= 2
            and self.image_data.shape[0] > 0
            and self.image_data.shape[1] > 0
        )

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        이미지 데이터 제외한 메타데이터만 반환 (XCom용)

        XCom은 JSON 직렬화 가능한 데이터만 전달 가능하므로,
        이미지 데이터는 별도 파일로 저장하고 메타데이터만 전달합니다.
        """
        return {
            'image_id': self.image_id,
            'camera_name': self.camera_name,
            'bay_id': self.bay_id,
            'source_path': self.source_path,
            'captured_at': self.captured_at.isoformat(),
            'metadata': self.metadata,
        }

    @classmethod
    def from_metadata_dict(
        cls,
        meta: Dict[str, Any],
        image_data: np.ndarray
    ) -> "CapturedImage":
        """
        메타데이터 딕셔너리와 이미지 데이터로 생성

        Args:
            meta: to_metadata_dict()로 생성된 메타데이터
            image_data: 이미지 데이터 (NumPy 배열)

        Returns:
            CapturedImage: 복원된 CapturedImage 객체
        """
        return cls(
            image_id=meta['image_id'],
            camera_name=meta['camera_name'],
            bay_id=meta['bay_id'],
            image_data=image_data,
            captured_at=datetime.datetime.fromisoformat(meta['captured_at']),
            source_path=meta['source_path'],
            metadata=meta.get('metadata', {}),
        )


@dataclass
class SceneState:
    """특정 시점의 현장 상태를 나타내는 데이터 클래스"""
    timestamp: str
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    assembly_status: str = "unknown"


@dataclass
class Event:
    """탐지된 이벤트를 나타내는 데이터 클래스"""
    event_type: Literal["object_appeared", "object_disappeared"]
    description: str
    timestamp: str


@dataclass
class FrameData:
    """
    단일 카메라 프레임의 모든 정보를 담는 DTO

    Observation에서 추론 결과를 담아 Fusion으로 전달합니다.

    Attributes:
        image_id: 이미지 고유 ID
        camera_name: 카메라 이름
        image_path: 이미지 파일 경로
        captured_at: 캡처 시각 (ISO 형식 문자열)
        image_shape: 이미지 shape (height, width)
        detections: YOLO 탐지 결과 리스트
        boundary_masks: 경계선 마스크 리스트 (NumPy 배열)
        pinjig_masks: 핀지그 마스크 리스트 (NumPy 배열)
        pinjig_classifications: 핀지그 분류 정보 리스트
        assembly_classifications: 조립 상태 분류 정보 리스트
    """
    image_id: str
    camera_name: str
    image_path: str
    captured_at: str
    image_shape: Tuple[int, int]
    detections: List[Dict[str, Any]] = field(default_factory=list)
    boundary_masks: List[np.ndarray] = field(default_factory=list)
    pinjig_masks: List[np.ndarray] = field(default_factory=list)
    pinjig_classifications: List[Dict[str, Any]] = field(default_factory=list)
    assembly_classifications: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ARMarkerData:
    """
    AR 마커 인식 및 검색 관련 데이터를 담는 데이터 클래스

    Attributes:
        image_id: 이미지 고유 ID
        image_path: 이미지 파일 경로
        camera_name: 카메라 이름
        camera_angle: 카메라 촬영 각도 (예: "snapshot_upimage", "snapshot_downimage")
        ar_log_search_start_time: DB AR 마커 인식 이력 검색 시작시간
        ar_log_search_end_time: DB AR 마커 인식 이력 검색 종료시간
        ar_boxes: 카메라 화각 내 AR 마커 정보 리스트
    """
    image_id: str
    image_path: str
    camera_name: str
    camera_angle: str
    ar_log_search_start_time: datetime.datetime
    ar_log_search_end_time: datetime.datetime
    ar_boxes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class InferenceResult:
    """
    AI 추론 결과 데이터 클래스

    Attributes:
        image_id: 원본 이미지 ID
        camera_name: 카메라 이름
        bay_id: Bay 식별자
        detections: YOLO 객체 탐지 결과 리스트
            - class_name: 클래스 이름
            - confidence: 신뢰도
            - bbox: [x1, y1, x2, y2]
        boundary_masks: SAM 경계선 마스크 리스트 (NumPy 배열)
        pinjig_masks: 핀지그 마스크 리스트 (NumPy 배열)
        pinjig_classifications: 핀지그 분류 정보 리스트
        assembly_classifications: 조립 상태 분류 정보 리스트
        processing_time: 처리 시간 (초)
        error_message: 에러 메시지 (실패 시)
    """
    image_id: str
    camera_name: str
    bay_id: str
    detections: List[Dict[str, Any]] = field(default_factory=list)
    boundary_masks: List[np.ndarray] = field(default_factory=list)
    pinjig_masks: List[np.ndarray] = field(default_factory=list)
    pinjig_classifications: List[Dict[str, Any]] = field(default_factory=list)
    assembly_classifications: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        """추론 성공 여부"""
        return self.error_message is None

    def to_frame_data(self, image_path: str, captured_at: str, image_shape: Tuple[int, int]) -> FrameData:
        """
        FrameData로 변환

        Args:
            image_path: 이미지 파일 경로
            captured_at: 캡처 시각 (ISO 형식 문자열)
            image_shape: 이미지 shape (height, width)

        Returns:
            FrameData: Fusion 모듈용 프레임 데이터
        """
        return FrameData(
            image_id=self.image_id,
            camera_name=self.camera_name,
            image_path=image_path,
            captured_at=captured_at,
            image_shape=image_shape,
            detections=self.detections,
            boundary_masks=self.boundary_masks,
            pinjig_masks=self.pinjig_masks,
            pinjig_classifications=self.pinjig_classifications,
            assembly_classifications=self.assembly_classifications,
        )
