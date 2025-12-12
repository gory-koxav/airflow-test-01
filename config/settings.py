# -*- coding: utf-8 -*-
"""
프로젝트 설정 파일

이 파일은 YAML 설정 파일을 로드하고 Pydantic 모델로 검증합니다.
Legacy visionflow-legacy-observation의 settings.py를 기반으로 하며,
Airflow 환경에 맞게 수정되었습니다.

주요 변경사항:
- Redis 설정 제거 (Airflow XCom으로 대체)
- Airflow 스케줄링 설정 추가
- 배포 관련 설정 추가 (bay_id, network 등)
- 카메라 ONVIF 연결 정보 추가
"""
import os
import yaml
import numpy as np

from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from pydantic.fields import PrivateAttr


### ------------------------------------------------------- ###
### -------------------- PATH SETTINGS -------------------- ###
### ------------------------------------------------------- ###
class PathSetting(BaseModel):
    """
    경로 설정

    모델 체크포인트, 출력 디렉토리 등의 경로를 관리합니다.
    None으로 설정된 경로는 base 경로를 기준으로 자동 resolve됩니다.
    """
    # 모델/출력 파일의 기본 경로
    base: Path = Field(default=Path("/opt/airflow/data"))
    # 원본 이미지가 저장된 경로
    image_data: Path = Field(default=Path("/opt/airflow/data/NFS_images"))

    # -------------------- CHECKPOINT -------------------- #
    # SAM checkpoint 경로
    sam_ckpt: Optional[Path] = None
    # SAM으로 분할된 이미지를 분류 YOLO Checkpoint 경로
    yolo_cls_ckpt: Optional[Path] = None
    # 객체 탐지 YOLO Checkpoint 경로
    yolo_od_ckpt: Optional[Path] = None
    # Assembly 분류 YOLO Checkpoint 경로
    yolo_asbl_ckpt: Optional[Path] = None

    # ---------------------- RESULT ---------------------- #
    # SAM 출력 경로
    sam_output: Optional[Path] = None
    # 경계선 탐지 출력 경로
    boundary_output: Optional[Path] = None
    # 투영 결과 출력 경로
    projection_output: Optional[Path] = None
    # 시각화 출력 경로
    visualize_output: Optional[Path] = None
    # 블록 높이 보정 출력 경로
    compensator_output: Optional[Path] = None

    @model_validator(mode="after")
    def resolve_ckpt_paths(self):
        """None인 경로들을 base 경로 기준으로 자동 resolve"""
        base = self.base.resolve()
        ckpt_dir = base / "checkpoints"
        out_dir = base / "output"

        # Checkpoint 경로 resolve
        self.sam_ckpt = self.sam_ckpt or (ckpt_dir / "segmentation_sam" / "sam_vit_h_4b8939.pth")
        self.yolo_cls_ckpt = self.yolo_cls_ckpt or (
            ckpt_dir / "segmentation_yolo_cls" / "best_250724.pt"
        )
        self.yolo_od_ckpt = self.yolo_od_ckpt or (
            ckpt_dir / "objectdetection_yolo" / "best_250408.pt"
        )
        self.yolo_asbl_ckpt = self.yolo_asbl_ckpt or (ckpt_dir / "stage_cls" / "best_250312.pt")

        # Output 경로 resolve (경로 충돌 제거)
        self.sam_output = (self.sam_output or (out_dir / "sam")).resolve()
        self.boundary_output = (self.boundary_output or (out_dir / "boundary")).resolve()
        self.projection_output = (self.projection_output or (out_dir / "projection")).resolve()
        self.visualize_output = (self.visualize_output or (out_dir / "visualize")).resolve()
        self.compensator_output = (
            self.compensator_output or (out_dir / "projection" / "compensator" / "block_height")
        ).resolve()

        return self


def ensure_directories(setting: PathSetting) -> None:
    """출력 디렉토리 자동 생성"""
    for p in [
        setting.sam_output,
        setting.boundary_output,
        setting.projection_output,
        setting.visualize_output,
        setting.compensator_output,
    ]:
        if p:
            os.makedirs(p, exist_ok=True)


### ------------------------------------------------------ ###
### ------------------- MODEL SETTINGS ------------------- ###
### ------------------------------------------------------ ###
class SamSetting(BaseModel):
    """SAM (Segment Anything Model) 설정"""
    # 사용할 SAM 모델의 타입 (예: "vit_h", "vit_l", "vit_b")
    type: str = "vit_h"
    # SAM으로 지그류를 탐지 가능성을 높이기 위해 이미지 상단을 회색으로 채우는 비율
    cutoff_pct_top: float = 0
    # SAM으로 지그류를 탐지 가능성을 높이기 위해 이미지 하단을 회색으로 채우는 비율
    cutoff_pct_bot: float = 0


class PipelineItem(BaseModel):
    """추론 파이프라인 아이템"""
    name: str
    module: str
    args: Dict = {}

    def get(self, key: str, default=None):
        """딕셔너리처럼 접근 가능하도록 지원"""
        data = self.model_dump()
        return data.get(key, default)


class ModelSetting(BaseModel):
    """모델 설정"""
    sam: SamSetting = Field(default_factory=SamSetting)
    pipeline: List[PipelineItem] = Field(
        default=[
            PipelineItem(name="YOLOObjectDetector", module="observation.inference.strategies"),
            PipelineItem(name="AutomaticSegmenter", module="observation.inference.strategies"),
            PipelineItem(name="MaskClassifier", module="observation.inference.strategies"),
            PipelineItem(
                name="SAMObjectBoundarySegmenter", module="observation.inference.strategies"
            ),
        ]
    )


### -------------------------------------------------------- ###
### ------------------ ALGORITHM SETTINGS ------------------ ###
### -------------------------------------------------------- ###
# -------------------- COMPENSATOR -------------------- #
class BlockHeightCompSet(BaseModel):
    """블록 높이 보정 설정"""
    # 이미지 높이의 최소 비율 (20%)
    min_height_ratio: float = 0.2
    # 이미지 높이의 최대 비율 (80%)
    max_height_ratio: float = 0.80
    # 조정 후 최소 높이 비율 (40%)
    adjust_min_height_ratio: float = 0.4
    # 조정 후 최대 높이 비율 (95%)
    adjust_max_height_ratio: float = 0.95

    # 이미지 너비의 최소 비율 (5%)
    min_width_ratio: float = 0.05
    # 이미지 너비의 최대 비율 (95%)
    max_width_ratio: float = 0.95

    # 좌우 마진 비율 (20%)
    side_margin_ratio: float = 0.2
    # 너비 조건 충족 시 좌우 마진 비율 (10%)
    width_margin_ratio: float = 0.1


class CompensatorSetting(BaseModel):
    """보정 알고리즘 설정"""
    block_height: BlockHeightCompSet = Field(default_factory=BlockHeightCompSet)


# -------------------- RESULT -------------------- #
class ThresholdSet(BaseModel):
    """임계값 설정"""
    # 블록 마스크와 객체 Bbox의 겹침 비율 임계값
    # (겹침 영역 넓이 / 객체 Bbox 넓이)
    overlap_object: float = 0.5
    # 블록 마스크와 AR 마커 Bbox의 겹침 비율 임계값
    # (겹침 영역 넓이 / AR 마커 Bbox 넓이)
    overlap_ar: float = 0.9


class VisualizeSet(BaseModel):
    """시각화 설정"""
    show_warped_image: bool = True
    show_object_boxes: bool = True
    show_boundary_masks: bool = True
    show_pinjig_masks: bool = True
    show_assembly_labels: bool = True
    dpi: int = 150
    grid_interval: int = 5
    mask_color: List[int] = [255, 0, 0]  # RGB
    mask_alpha: float = 0.4
    pinjig_mask_alpha: float = 0.4
    box_color: str = "magenta"
    box_linewidth: float = 1.5


class ResultSetting(BaseModel):
    """결과 처리 설정"""
    threshold: ThresholdSet = Field(default_factory=ThresholdSet)
    visualize: VisualizeSet = Field(default_factory=VisualizeSet)


class AlgorithmSetting(BaseModel):
    """알고리즘 설정"""
    compensator: CompensatorSetting = Field(default_factory=CompensatorSetting)
    result: ResultSetting = Field(default_factory=ResultSetting)


### --------------------------------------------------------- ###
### -------------------- TARGET SETTINGS -------------------- ###
### --------------------------------------------------------- ###
class ClassSetting(BaseModel):
    """대상 클래스 설정"""
    # 경계선 탐지 대상 클래스
    boundary: List[str] = Field(default=["panel", "block", "HP_block"])
    # 투영 대상 클래스
    projection: List[str] = Field(
        default=["panel", "block", "HP_block", "crane", "grinder", "welder", "step_board", "pipe"]
    )
    # 핀지그 대상 클래스
    pinjig: List[str] = Field(
        default=["pinjig", "hbeamjig", "gridjig", "pinjig_empty", "gridjig_empty", "hbeamjig_empty", "flatjig_empty"]
    )

    # 클래스 이름 -> 인덱스 매핑 (자동 생성)
    boundary_name2idx: Optional[Dict[str, int]] = None
    projection_name2idx: Optional[Dict[str, int]] = None
    pinjig_name2idx: Optional[Dict[str, int]] = None

    @model_validator(mode="after")
    def resolve_name2idx(self):
        """클래스 이름 -> 인덱스 매핑 자동 생성"""
        self.boundary_name2idx = {name: idx for idx, name in enumerate(self.boundary)}
        self.projection_name2idx = {name: idx for idx, name in enumerate(self.projection)}
        self.pinjig_name2idx = {name: idx for idx, name in enumerate(self.pinjig)}

        return self


class TargetSetting(BaseModel):
    """대상 설정"""
    classes: ClassSetting = Field(default_factory=ClassSetting)
    # 필터링할 클래스 (예: crane은 투영 결과에서 제외)
    filter_classes: List[str] = Field(default=["crane"])
    # 블록과 매칭시킬 객체 클래스 및 결과 딕셔너리의 카운트 키 매핑
    matching: Dict[str, str] = Field(
        default={
            "grinder": "matched_grids_counts",
            "welder": "matched_weld_counts",
            "step_board": "matched_step_boards_counts",
            "pipe": "matched_pipes_counts",
        }
    )


### -------------------------------------------------------- ###
### ----------------------- SETTINGS ----------------------- ###
### -------------------------------------------------------- ###
class Setting(BaseModel):
    """
    메인 설정 클래스

    base_config.yaml에서 로드되는 공통 설정입니다.
    Bay별 설정은 BaySetting에서 별도로 관리됩니다.
    """
    # Bay 설정 파일 경로 (환경변수로 오버라이드 가능)
    bay_config_path: str = Field(default="config/bay/bay_12bay-west.yaml")

    path: PathSetting = Field(default_factory=PathSetting)
    model: ModelSetting = Field(default_factory=ModelSetting)
    algorithm: AlgorithmSetting = Field(default_factory=AlgorithmSetting)
    target: TargetSetting = Field(default_factory=TargetSetting)


### -------------------------------------------------------- ###
### --------------------- BAY SETTINGS --------------------- ###
### -------------------------------------------------------- ###

# -------------------- DEPLOYMENT -------------------- #
class NetworkSetting(BaseModel):
    """네트워크 설정"""
    # 카메라 서브넷
    camera_subnet: str = ""
    # Observer 서버 IP
    observer_ip: str = ""


class ScheduleSetting(BaseModel):
    """스케줄 설정 (Airflow Cron 표현식)"""
    # Observation DAG 스케줄
    observation: Optional[str] = "*/10 * * * *"
    # Fusion DAG 스케줄 (None이면 Trigger 기반)
    fusion: Optional[str] = None
    # Tracking DAG 스케줄 (None이면 Trigger 기반)
    tracking: Optional[str] = None


class DeploymentSetting(BaseModel):
    """배포 설정"""
    # Bay 식별자 (예: "12bay-west", "64bay")
    bay_id: str = ""
    # Bay 설명
    description: str = ""
    # 활성화 여부
    enabled: bool = True
    # 네트워크 설정
    network: NetworkSetting = Field(default_factory=NetworkSetting)
    # 스케줄 설정
    schedule: ScheduleSetting = Field(default_factory=ScheduleSetting)
    # 운영 시간 [(start_h, start_m, end_h, end_m), ...]
    operating_hours: List[List[int]] = Field(default=[[0, 0, 23, 59]])
    # 스킵 시간 [(skip_h, skip_m, tolerance), ...]
    skip_times: List[List[int]] = Field(default_factory=list)


# -------------------- FACTORY -------------------- #
class FactorySetting(BaseModel):
    """공장 레이아웃 설정"""
    bay: int = Field(default=64)
    # Bay 열 식별자 (예: ["D", "E"] 또는 ["C", "B"])
    bay_column: List[str] = Field(default=["D", "E"])
    # 컬럼 번호 -> 인덱스 매핑
    column2idx: Dict[int, int] = Field(default_factory=dict)
    # 공장 너비 (미터)
    width: float = Field(default=200)
    # 공장 높이 (미터)
    height: float = Field(default=30)
    # 열 1의 높이 오프셋 (예: D열)
    offset_height_sub_1: float = Field(default=4.5)
    # 열 2의 높이 오프셋 (예: E열)
    offset_height_sub_2: float = Field(default=-4.5)
    # 열 1의 팬 오프셋 (예: D열)
    offset_pan_sub_1: float = Field(default=180)
    # 열 2의 팬 오프셋 (예: E열)
    offset_pan_sub_2: float = Field(default=0)
    # 틸트 오프셋
    offset_tilt: float = Field(default=42)
    # 컬럼 오프셋
    offset_column: float = Field(default=5)
    # 카메라 높이 (미터)
    camera_height: float = Field(default=19)


# -------------------- CAMERA -------------------- #
class ExtrinsicInfo(BaseModel):
    """카메라 외부 파라미터"""
    # 위치 [x, y, z] (미터)
    translation: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    # 팬 오프셋 (도)
    offset_pan: float = 0.0
    # 틸트 오프셋 (도)
    offset_tilt: float = 0.0


class PTZInfo(BaseModel):
    """PTZ (Pan-Tilt-Zoom) 정보"""
    pan: int = 0
    tilt: int = 0
    zoom: int = 10


class CameraConnectionInfo(BaseModel):
    """카메라 ONVIF 연결 정보"""
    host: str = ""
    port: int = 80
    username: str = "admin"
    # password는 환경변수에서 로드 (보안)


class CameraInfoItem(BaseModel):
    """개별 카메라 정보"""
    bay: int
    column: str
    # ONVIF 연결 정보
    connection: CameraConnectionInfo = Field(default_factory=CameraConnectionInfo)
    # 외부 파라미터
    extrinsic: ExtrinsicInfo = Field(default_factory=ExtrinsicInfo)
    # 상단 촬영 PTZ 설정
    up: PTZInfo = Field(default_factory=PTZInfo)
    # 하단 촬영 PTZ 설정
    down: PTZInfo = Field(default_factory=PTZInfo)


class CameraParameterSetting(BaseModel):
    """카메라 내부 파라미터 설정"""
    # 초점 거리 x (fx)
    focal_x: float = 1450
    # 초점 거리 y (fy)
    focal_y: float = 1450
    # 주점 x좌표 (cx)
    center_x: float = 960
    # 주점 y좌표 (cy)
    center_y: float = 540

    # 왜곡 계수 (방사형 왜곡 k1, k2, 접선 왜곡 p1, p2, 방사형 왜곡 k3)
    dist_coeffs: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # 내부 파라미터 행렬 (자동 생성)
    intrinsics_list: Optional[List[List[float]]] = None
    _intrinsics_np: Optional[np.ndarray] = PrivateAttr(default=None)
    _dist_coeffs_np: Optional[np.ndarray] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def construct_matrices(self):
        """내부 파라미터 행렬 자동 구성"""
        if len(self.dist_coeffs) not in (4, 5, 8):
            raise ValueError(f"dist_coeffs length must be 4, 5, or 8. Got: {len(self.dist_coeffs)}")

        intrinsics_matrix = [
            [self.focal_x, 0.0, self.center_x],
            [0.0, self.focal_y, self.center_y],
            [0.0, 0.0, 1.0],
        ]

        self.intrinsics_list = intrinsics_matrix
        self._intrinsics_np = np.array(intrinsics_matrix, dtype=np.float32)
        self._dist_coeffs_np = np.array(self.dist_coeffs, dtype=np.float32)

        if self._intrinsics_np.shape != (3, 3):
            raise ValueError("내부 행렬 구성 오류: 3x3 형태가 아닙니다.")

        return self

    @property
    def intrinsics_np(self) -> np.ndarray:
        """내부 파라미터 행렬 (NumPy)"""
        return self._intrinsics_np

    @property
    def dist_coeffs_np(self) -> np.ndarray:
        """왜곡 계수 (NumPy)"""
        return self._dist_coeffs_np


class CameraControlSetting(BaseModel):
    """카메라 제어 설정"""
    pan_step_ratio: float = Field(default=0.8)
    tilt_step_ratio: float = Field(default=1.0)
    zoom_factor: float = Field(default=30)


class CameraSetting(BaseModel):
    """카메라 설정"""
    # 이미지 해상도
    resolution_x: int = Field(default=1920)
    resolution_y: int = Field(default=1080)

    # 내부 파라미터
    intrinsic: CameraParameterSetting = Field(default_factory=CameraParameterSetting)
    # 제어 파라미터
    control: CameraControlSetting = Field(default_factory=CameraControlSetting)

    # 카메라 이름 목록
    camera_list: List[str] = Field(default_factory=list)
    # 카메라별 상세 설정
    RAW: Dict[str, CameraInfoItem] = Field(default_factory=dict)

    # 내부 캐시 (투영/외부 파라미터)
    _projection_map: Dict[str, Dict[str, Tuple[int, int, int]]] = PrivateAttr(default_factory=dict)
    _extrinsic_map: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def _rotation_matrix(self, pan_deg: float, tilt_deg: float) -> np.ndarray:
        """팬/틸트 각도로 회전 행렬 계산"""
        pan = np.deg2rad(pan_deg)
        tilt = np.deg2rad(tilt_deg)

        R_tilt = np.array(
            [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]],
            dtype=np.float32,
        )

        R_pan = np.array(
            [[np.cos(pan), -np.sin(pan), 0], [np.sin(pan), np.cos(pan), 0], [0, 0, 1]],
            dtype=np.float32,
        )

        return R_pan @ R_tilt

    @model_validator(mode="after")
    def post_process_camera_data(self):
        """카메라 데이터 후처리 (투영 맵, 외부 파라미터 맵 구성)"""
        cam_list_set = set(self.camera_list)
        raw_set = set(self.RAW.keys())

        if cam_list_set != raw_set:
            missing_in_raw = cam_list_set - raw_set
            missing_in_list = raw_set - cam_list_set

            raise ValueError(
                f"CameraSetting: camera_list와 RAW의 key가 일치하지 않습니다. "
                f"RAW에 없는 camera_list 항목: {missing_in_raw or '없음'}, "
                f"camera_list에 없는 RAW 항목: {missing_in_list or '없음'}"
            )

        for name, cam in self.RAW.items():
            # Projection 맵 구성 (up/down PTZ를 튜플로 저장)
            self._projection_map[name] = {
                "up": (cam.up.pan, cam.up.tilt, cam.up.zoom),
                "down": (cam.down.pan, cam.down.tilt, cam.down.zoom),
            }

            R = self._rotation_matrix(
                pan_deg=cam.extrinsic.offset_pan, tilt_deg=cam.extrinsic.offset_tilt
            )

            t = np.array(cam.extrinsic.translation, dtype=np.float32)

            self._extrinsic_map[name] = {
                "R": R,
                "t": t,
            }

        return self

    def get_extrinsic(self, name: str) -> Optional[Dict[str, np.ndarray]]:
        """카메라 외부 파라미터 조회"""
        return self._extrinsic_map.get(name)

    def get_ptz(self, name: str, mode: str = "up") -> Optional[Tuple[int, int, int]]:
        """카메라 PTZ 설정 조회"""
        return self._projection_map.get(name, {}).get(mode)

    def get_camera_info(self, name: str) -> Optional[CameraInfoItem]:
        """카메라 정보 조회"""
        return self.RAW.get(name)

    def get_cameras_for_airflow(self) -> List[Dict[str, Any]]:
        """
        Airflow DAG에서 사용할 카메라 리스트 반환

        기존 bay_configs.py의 cameras 리스트 형식과 호환됩니다.
        """
        camera_password = os.getenv("CAMERA_PASSWORD", "")
        cameras = []
        for name in self.camera_list:
            cam_info = self.RAW.get(name)
            if cam_info:
                cameras.append({
                    "name": name,
                    "host": cam_info.connection.host,
                    "port": cam_info.connection.port,
                    "username": cam_info.connection.username,
                    "password": camera_password,
                })
        return cameras


# -------------------- FACTORY LAYOUT -------------------- #
class AreaSetting(BaseModel):
    """서브베이 영역 설정"""
    top_left: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    bottom_right: List[float] = Field(default_factory=lambda: [0.0, 0.0])


class FactoryLayoutSetting(BaseModel):
    """공장 레이아웃 설정"""
    width: float = 150
    height: float = 40
    vertical_center: float = 20
    areas: Dict[str, AreaSetting] = Field(default_factory=dict)


# -------------------- TRACKING -------------------- #
class TrackingSetting(BaseModel):
    """추적 설정"""
    iomin_threshold: float = 0.6
    contour_min_distance: float = 5.0
    exit_confirmation_threshold: float = 0.49
    hbeam_jig_subbays: List[str] = Field(default_factory=list)
    ignore_subbays: List[str] = Field(default_factory=list)


# -------------------- BAY -------------------- #
class BaySetting(BaseModel):
    """
    Bay별 설정 클래스

    각 Bay의 개별 설정을 담습니다.
    bay_*.yaml 파일에서 로드됩니다.
    """
    # 배포 설정 (bay_id, schedule 등)
    deployment: DeploymentSetting = Field(default_factory=DeploymentSetting)
    # 공장 설정
    factory: FactorySetting = Field(default_factory=FactorySetting)
    # 카메라 설정
    camera: CameraSetting = Field(default_factory=CameraSetting)
    # 공장 레이아웃 (서브베이 영역)
    factory_layout: FactoryLayoutSetting = Field(default_factory=FactoryLayoutSetting)
    # 추적 설정
    tracking: TrackingSetting = Field(default_factory=TrackingSetting)


### ----------------------------------------- ###
###            Setting Loader                 ###
### ----------------------------------------- ###
def load_setting(config_path: Optional[Path] = None) -> Setting:
    """
    메인 설정 로드

    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)

    Returns:
        Setting: 메인 설정 객체
    """
    if config_path is None:
        root = Path(__file__).resolve().parent
        config_path = root / "base_config.yaml"

    if not config_path.exists():
        # 설정 파일이 없으면 기본값 사용
        return Setting()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Setting(**raw) if raw else Setting()


def load_bay_setting(path: Union[str, Path]) -> BaySetting:
    """
    Bay 설정 로드

    Args:
        path: Bay 설정 파일 경로 (상대 경로 또는 절대 경로)

    Returns:
        BaySetting: Bay 설정 객체
    """
    if isinstance(path, str):
        path = Path(path)

    # 상대 경로인 경우 프로젝트 루트 기준으로 변환
    if not path.is_absolute():
        root = Path(__file__).resolve().parent.parent  # config/ 상위 = 프로젝트 루트
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Bay config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return BaySetting(**raw)


def get_setting() -> Setting:
    """
    전역 설정 반환 (싱글톤)

    환경변수 AIRFLOW_CONFIG_PATH로 설정 파일 경로 오버라이드 가능
    """
    config_path = os.getenv("AIRFLOW_CONFIG_PATH")
    if config_path:
        return load_setting(Path(config_path))
    return load_setting()


def get_bay_setting(bay_config_path: Optional[str] = None) -> BaySetting:
    """
    Bay 설정 반환

    Args:
        bay_config_path: Bay 설정 파일 경로 (None이면 환경변수 또는 기본 설정 사용)

    환경변수:
        AIRFLOW_BAY_CONFIG_PATH: Bay 설정 파일 경로
    """
    if bay_config_path is None:
        bay_config_path = os.getenv("AIRFLOW_BAY_CONFIG_PATH")

    if bay_config_path is None:
        # 메인 설정에서 bay_config_path 읽기
        setting = get_setting()
        bay_config_path = setting.bay_config_path

    return load_bay_setting(bay_config_path)


### ----------------------------------------- ###
###       DAG Factory Helper Functions        ###
### ----------------------------------------- ###

def get_observer_queue(bay_id: str) -> str:
    """Observer Queue 이름 생성"""
    return f"obs_{bay_id}"


def get_processor_queue(bay_id: str) -> str:
    """Processor Queue 이름 생성"""
    return f"proc_{bay_id}"


def get_image_provider_config(bay_id: str) -> Dict[str, Any]:
    """
    Bay별 image_provider_config 반환

    PathSetting의 image_data 경로를 기준으로 Bay별 경로를 생성합니다.
    """
    setting = get_setting()
    base_path = setting.path.image_data / bay_id
    return {
        "base_path": str(base_path),
        "time_tolerance_hours": 100000.0,
    }


def get_enabled_bay_settings() -> Dict[str, BaySetting]:
    """
    활성화된 모든 Bay 설정 반환

    config/bay/ 디렉토리의 모든 bay_*.yaml 파일을 로드하고,
    enabled=True인 Bay만 BaySetting 객체로 반환합니다.

    Returns:
        Dict[bay_id, BaySetting]: Bay ID를 키로 하는 BaySetting 딕셔너리
    """
    import logging
    logger = logging.getLogger(__name__)

    bay_settings = {}
    config_dir = Path(__file__).parent / "bay"

    if not config_dir.exists():
        logger.warning(f"Bay 설정 디렉토리가 없습니다: {config_dir}")
        return {}

    for yaml_file in config_dir.glob("bay_*.yaml"):
        try:
            bay_setting = load_bay_setting(yaml_file)

            if not bay_setting.deployment.enabled:
                continue

            bay_id = bay_setting.deployment.bay_id
            if not bay_id:
                logger.warning(f"bay_id가 없는 설정 파일: {yaml_file}")
                continue

            bay_settings[bay_id] = bay_setting
            logger.info(f"Bay 설정 로드 완료: {bay_id}")

        except Exception as e:
            logger.error(f"Bay 설정 로드 실패: {yaml_file}, 에러: {e}")

    return bay_settings


### ----------------------------------------- ###
###         Maintenance Config                ###
### ----------------------------------------- ###

MAINTENANCE_CONFIG = {
    "data_retention": {
        # 파일 타입별 보관 기간 (일 단위)
        "pkl_files": 3,
        "json_files": 7,
        "image_files": 365,  # 원본 이미지 1년 보관

        # 모듈별 결과물 보관 기간
        "observation_results": 7,
        "fusion_results": 7,
        "tracking_results": 30,  # 추적 이력은 길게 보관
    },

    "log_retention": {
        "airflow_task_logs": 30,
        "airflow_scheduler_logs": 7,
        "celery_worker_logs": 7,
    },

    # 스케줄 설정
    "schedules": {
        "data_cleanup": "0 2 * * 0",    # 매주 일요일 2AM
        "log_cleanup": "0 3 * * *",     # 매일 3AM
    },

    # 안전 장치
    "safety": {
        "min_free_space_gb": 10,  # 최소 여유 공간 (GB)
        "dry_run": False,          # True면 삭제하지 않고 로그만
    },
}


def get_maintenance_config() -> Dict[str, Any]:
    """유지보수 설정 반환 (환경별 오버라이드 가능)"""
    config = MAINTENANCE_CONFIG.copy()

    # 환경 변수로 오버라이드 (선택적)
    if pkl_days := os.getenv("MAINTENANCE_PKL_RETENTION_DAYS"):
        config["data_retention"]["pkl_files"] = int(pkl_days)

    if log_days := os.getenv("MAINTENANCE_LOG_RETENTION_DAYS"):
        config["log_retention"]["airflow_task_logs"] = int(log_days)

    if dry_run := os.getenv("MAINTENANCE_DRY_RUN"):
        config["safety"]["dry_run"] = dry_run.lower() in ("true", "1", "yes")

    return config
