# Observation 마이그레이션 전략

## 0. 용어 정의 및 명명 규칙

> ⚠️ **중요**: 본 문서 및 모든 코드에서 아래 명명 규칙을 일관되게 준수해야 합니다.

### 0.1 핵심 용어 체계

| 카테고리 | 작업(Task/Process) | 실행 주체(Worker/Module) | 설명 |
|----------|-------------------|------------------------|------|
| 카메라 관측 | **observation** | **observer** | 카메라 이미지 획득 및 AI 추론 |
| 다중 카메라 융합 | **fusion** | **fuser** | 다중 카메라 데이터 융합 분석 |
| 이동 추적 | **tracking** | **tracker** | 블록 이동 추적 및 기록 |

### 0.2 명명 규칙 상세

#### DAG 명명
```
{task_type}_{bay_id}_dag

예시:
- observation_3bay_north_dag  ✅
- fusion_3bay_north_dag       ✅
- tracking_3bay_north_dag     ✅
- tracker_3bay_north_dag      ❌ (잘못된 예)
```

#### Queue 명명
```
Observer Queue:  obs_{bay_id}    (예: obs_3bay_north)
Processor Queue: proc_{bay_id}   (예: proc_3bay_north)
```

#### Worker 명명
```
{role}-{bay_id}

Observer Worker:  observer-{bay_id}   (예: observer-3bay_north)
Processor Worker: processor-{bay_id}  (예: processor-3bay_north)

※ Processor는 fusion과 tracking을 모두 처리하는 통합 Worker
```

#### Task ID 명명
```
{동사}_{명사} 형식

예시 (Observation DAG 내 Task들):
- capture_images          (이미지 캡처)
- check_should_proceed    (수행 여부 판단)
- run_inference           (AI 추론)
- trigger_fusion          (Fusion DAG 트리거)

예시 (Fusion DAG 내 Task들):
- process_fusion          (융합 처리)
- trigger_tracking        (Tracking DAG 트리거)

예시 (Tracking DAG 내 Task들):
- process_tracking        (추적 처리)
```

#### 함수 명명
```
run_{task_type}_task 또는 run_{action}_task

예시:
- run_capture_task()              ✅
- run_inference_task()            ✅
- check_execution_conditions()    ✅
- run_fusion_task()               ✅
- run_tracking_task()             ✅
- run_tracker_task()              ❌ (잘못된 예)
```

#### 태그 명명
```
DAG 태그는 task type을 사용:
- ['observation', bay_id, ...]
- ['fusion', bay_id, ...]
- ['tracking', bay_id, ...]
```

### 0.3 혼동하기 쉬운 용어 정리

| 잘못된 사용 | 올바른 사용 | 비고 |
|------------|------------|------|
| tracker DAG | tracking DAG | DAG/Task는 작업명 사용 |
| observer DAG | observation DAG | DAG/Task는 작업명 사용 |
| fuser DAG | fusion DAG | DAG/Task는 작업명 사용 |
| tracking worker | tracker (또는 processor) | Worker는 주체명 사용 |
| observation worker | observer | Worker는 주체명 사용 |

### 0.4 코드 레벨 체크리스트

마이그레이션 구현 시 아래 항목을 확인하세요:

```python
# ✅ 올바른 예시 - DAG/함수 명명
dag_id = f"tracking_{bay_id}_dag"
def run_tracking_task(): ...
tags = ['tracking', bay_id, 'processor']

# ❌ 잘못된 예시
dag_id = f"tracker_{bay_id}_dag"
def run_tracker_task(): ...
tags = ['tracker', bay_id, 'processor']
```

```python
# ✅ 올바른 예시 - Observation DAG Task 흐름
# 모든 Task는 Observer Queue에서 실행 (Bay 물리 서버)
capture_images >> check_should_proceed >> run_inference >> trigger_fusion

# ❌ 잘못된 예시 - Main Server에서 조건 체크
check_schedule >> observe_and_inference  # check가 Main Server에서 실행되면 안됨
```

```python
# ✅ 올바른 예시 - 데이터 경로
result_path = f"/opt/airflow/data/observation/{bay_id}/{batch_id}/result.pkl"
OBSERVATION_DATA_PATH = "/opt/airflow/data/observation"

# ❌ 잘못된 예시 - 이전 pickle 경로
result_path = f"/opt/airflow/data/pickle/{bay_id}/{batch_id}/result.pkl"
PICKLE_DATA_PATH = "/opt/airflow/data/pickle"
```

---

## 1. 개요

### 1.1 마이그레이션 목표
- 기존 `producer_main.py`의 while 루프 기반 스케줄링을 Airflow DAG 기반으로 전환
- Redis pub-sub 기반 데이터 전달을 XCom + Pickle 파일 기반으로 전환
- **다중 Bay 지원**: 3bay_north, 4bay, 5bay 등 여러 Bay로 즉시 스케일업 가능한 구조
- **물리적 격리**: Bay별 Observer가 해당 Bay의 카메라만 접근 가능한 네트워크 구조 지원
- **중앙 집중 처리**: Fusion/Tracking은 Main Server에서 통합 처리

### 1.2 용어 정의
| 기존 용어 | 신규 용어 | 설명 |
|----------|----------|------|
| Producer | Observation | 카메라 관측 및 AI 추론 수행 모듈 |
| producer_main.py | observation_dag.py | Airflow DAG로 마이그레이션 |
| CaptureService | ImageProvider | 이미지 획득 (Strategy Pattern) |
| Situation Awareness | Fusion | 다중 카메라 융합 분석 모듈 |
| Worker (observation) | Observer | Bay별 Observation 전용 Worker |

### 1.3 현재 시스템 구조 (AS-IS)
```
┌─────────────────────────────────────────────────────────────┐
│ producer_main.py (Standalone Process)                       │
│                                                             │
│   while True:                                               │
│     1. is_in_operating_hours() 체크                         │
│     2. should_skip_execution() 체크                         │
│     3. generate_batch_id() 생성                             │
│     4. capture_images() - 12개 ONVIF 카메라 (3bay_north)    │
│     5. ai_inference() - YOLO + SAM + Classification         │
│     6. save_batch_results() → Redis 저장 + Publish          │
│     7. wait_until_next_execution()                          │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ Redis: "cctv_image_state_updates"
                           │ Payload: {batch_id, status, count}
                           ▼
                  [Fusion (구 Situation Awareness)]
                           │
                           ▼
                      [Tracker]
```

### 1.4 목표 시스템 구조 (TO-BE)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Central Server (Main Server Host)                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Airflow Main: Scheduler + API Server + PostgreSQL + Redis         │  │
│  │                                                                   │  │
│  │ DAGs:                                                             │  │
│  │  - observation_3bay_north_dag  → queue: obs_3bay_north            │  │
│  │  - observation_4bay_dag        → queue: obs_4bay                  │  │
│  │  - observation_5bay_dag        → queue: obs_5bay                  │  │
│  │  - fusion_3bay_north_dag       → queue: proc_3bay_north           │  │
│  │  - fusion_4bay_dag             → queue: proc_4bay                 │  │
│  │  - tracking_3bay_north_dag     → queue: proc_3bay_north           │  │
│  │  - tracking_4bay_dag           → queue: proc_4bay                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Processing Workers (Same Host as Main Server)                     │  │
│  │                                                                   │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐     │  │
│  │  │ Processor       │ │ Processor       │ │ Processor       │     │  │
│  │  │ 3bay_north      │ │ 4bay            │ │ 5bay            │     │  │
│  │  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤     │  │
│  │  │Queue:           │ │Queue:           │ │Queue:           │     │  │
│  │  │proc_3bay_north  │ │proc_4bay        │ │proc_5bay        │     │  │
│  │  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤     │  │
│  │  │Tasks:           │ │Tasks:           │ │Tasks:           │     │  │
│  │  │- fusion         │ │- fusion         │ │- fusion         │     │  │
│  │  │- tracking       │ │- tracking       │ │- tracking       │     │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ Celery Task Dispatch (by Queue)
           ┌──────────────────────┼──────────────────────┐
           ▼                      ▼                      ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  3bay_north Zone    │ │     4bay Zone       │ │     5bay Zone       │
│  (Physical Server A)│ │ (Physical Server B) │ │ (Physical Server C) │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ Observer:           │ │ Observer:           │ │ Observer:           │
│ observer-3bay_north │ │ observer-4bay       │ │ observer-5bay       │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ Queue:              │ │ Queue:              │ │ Queue:              │
│ obs_3bay_north      │ │ obs_4bay            │ │ obs_5bay            │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ Camera Network:     │ │ Camera Network:     │ │ Camera Network:     │
│ 10.150.160.x        │ │ 10.150.161.x        │ │ 10.150.162.x        │
│ (12 cameras)        │ │ (N cameras)         │ │ (N cameras)         │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ Task:               │ │ Task:               │ │ Task:               │
│ - observation ONLY  │ │ - observation ONLY  │ │ - observation ONLY  │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ ⛔ Cannot access    │ │ ⛔ Cannot access    │ │ ⛔ Cannot access    │
│ other bay cameras   │ │ other bay cameras   │ │ other bay cameras   │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

---

## 2. 물리적 서버 구성 및 네트워크 격리

### 2.1 운영 환경 물리적 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SAMHO Factory Network                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Central Server Room                           │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Main Server Host                                        │    │   │
│  │  │  ┌─────────────────────────────────────────────────┐    │    │   │
│  │  │  │ Airflow Main (Scheduler, API, PostgreSQL, Redis)│    │    │   │
│  │  │  └─────────────────────────────────────────────────┘    │    │   │
│  │  │  ┌─────────────────────────────────────────────────┐    │    │   │
│  │  │  │ Processing Workers (Fusion + Tracking)          │    │    │   │
│  │  │  │  - processor-3bay_north (queue: proc_3bay_north)│    │    │   │
│  │  │  │  - processor-4bay       (queue: proc_4bay)      │    │    │   │
│  │  │  │  - processor-5bay       (queue: proc_5bay)      │    │    │   │
│  │  │  └─────────────────────────────────────────────────┘    │    │   │
│  │  │  - IP: 10.150.150.1 (Management Network)                │    │   │
│  │  │  - 모든 Observer와 통신 가능                             │    │   │
│  │  │  - 공유 스토리지: 별도 NFS 파일 서버 마운트              │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │   3Bay North     │  │      4Bay        │  │      5Bay        │      │
│  │   Zone           │  │      Zone        │  │      Zone        │      │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤      │
│  │ Observer Server  │  │ Observer Server  │  │ Observer Server  │      │
│  │ IP: 10.150.160.1 │  │ IP: 10.150.161.1 │  │ IP: 10.150.162.1 │      │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤      │
│  │ Task:            │  │ Task:            │  │ Task:            │      │
│  │ observation ONLY │  │ observation ONLY │  │ observation ONLY │      │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤      │
│  │ Camera Network   │  │ Camera Network   │  │ Camera Network   │      │
│  │ 10.150.160.x     │  │ 10.150.161.x     │  │ 10.150.162.x     │      │
│  │ (12 cameras)     │  │ (N cameras)      │  │ (N cameras)      │      │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤      │
│  │ ⛔ Cannot access │  │ ⛔ Cannot access │  │ ⛔ Cannot access │      │
│  │ other bay cams   │  │ other bay cams   │  │ other bay cams   │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Worker 역할 분리

| Worker 유형 | 명명 규칙 | 위치 | 담당 Task | Queue |
|-------------|-----------|------|-----------|-------|
| **Observer** | `observer-{bay_id}` | Bay별 물리 서버 | observation | `obs_{bay_id}` |
| **Processor** | `processor-{bay_id}` | Main Server Host | fusion, tracking | `proc_{bay_id}` |

### 2.3 네트워크 격리 원칙

| 원칙 | 설명 |
|------|------|
| **Bay별 네트워크 분리** | 각 Bay의 카메라는 해당 Bay 전용 서브넷에 위치 |
| **Observer 접근 제한** | Observer는 자신이 담당하는 Bay의 카메라만 네트워크 접근 가능 |
| **Central 통신** | 모든 Observer/Processor는 Main Server(PostgreSQL, Redis)와 통신 |
| **Processor 카메라 접근 불가** | Processor는 카메라에 직접 접근하지 않음 (Pickle 데이터만 처리) |

### 2.4 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Flow                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Bay Zone (Physical Server)              Shared Storage (NFS File Server) │
│  ┌──────────────────────┐               ┌──────────────────────────┐   │
│  │ Observer             │               │ NFS File Server          │   │
│  │                      │   ────────▶   │ /exports/airflow-data/   │   │
│  │ 1. Capture Images    │   Pickle      │   observation/           │   │
│  │ 2. Check Conditions  │   Save        │     {bay_id}/{batch_id}/ │   │
│  │ 3. AI Inference      │               │       result.pkl         │   │
│  │ 4. Save to Pickle    │               │                          │   │
│  │ 5. Return XCom       │               │ ※ 개발환경: ./data 폴더  │   │
│  └──────────────────────┘               └──────────────────────────┘   │
│                                                    │                    │
│                                                    ▼ NFS Mount          │
│                                         ┌──────────────────────────┐   │
│                                         │ Processor (Central)      │   │
│                                         │                          │   │
│                                         │ 1. Read Pickle (XCom)    │   │
│                                         │ 2. Fusion Processing     │   │
│                                         │ 3. Tracking              │   │
│                                         │ 4. Save Results          │   │
│                                         └──────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Worker 구성 전략

### 3.1 Processor Worker 구성 옵션 비교

| 옵션 | 구성 | 장점 | 단점 |
|------|------|------|------|
| **A. 단일 Worker** | 모든 Bay의 fusion+tracking을 1개 Worker에서 | 간단, 리소스 효율적 | 병목 가능, SPOF |
| **B. Bay별 Worker** | Bay별로 fusion+tracking 전용 Processor | 격리, 병렬처리, 스케일업 용이 | 리소스 증가 |
| **C. 역할별 Worker** | fusion Worker 1개 + tracking Worker 1개 | 역할 분리, 튜닝 용이 | Bay간 영향, 복잡 |

### 3.2 권장: Bay별 Processor Worker (옵션 B)

**권장 이유:**
1. **격리성**: 한 Bay의 처리 지연이 다른 Bay에 영향 없음
2. **스케일업 용이**: Bay 추가 시 Processor Worker만 추가
3. **장애 격리**: 특정 Bay Processor 장애 시 다른 Bay 정상 운영
4. **부하 분산**: Bay별 처리량 불균형에 대응 가능
5. **일관성**: Observer와 동일한 Bay별 분리 구조 유지

**초기 운영 전략:**
- Bay가 1-2개일 때: 단일 Worker로 시작 가능 (옵션 A)
- Bay가 3개 이상: Bay별 Processor로 전환 (옵션 B)

### 3.3 Queue 명명 규칙

```
Queue 명명 패턴:
- Observer Queue:  obs_{bay_id}     (예: obs_3bay_north)
- Processor Queue: proc_{bay_id}    (예: proc_3bay_north)

Worker-Queue 매핑:
┌─────────────────────────────────────────────────────────────────────────┐
│ Observer (Bay 물리 서버)                                                │
│  observer-3bay_north  ←→  Queue: obs_3bay_north                        │
│  observer-4bay        ←→  Queue: obs_4bay                              │
│  observer-5bay        ←→  Queue: obs_5bay                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Processor (Main Server Host)                                            │
│  processor-3bay_north ←→  Queue: proc_3bay_north                       │
│  processor-4bay       ←→  Queue: proc_4bay                             │
│  processor-5bay       ←→  Queue: proc_5bay                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 전체 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   3bay_north Pipeline                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Bay Zone (Physical Server A)          Central Server                   │
│  ┌──────────────────────┐             ┌──────────────────────────────┐ │
│  │   Observation        │             │       Fusion                  │ │
│  │   3bay_north         │ ──────────▶ │       3bay_north              │ │
│  ├──────────────────────┤  Trigger    ├──────────────────────────────┤ │
│  │ queue: obs_3bay_north│             │ queue: proc_3bay_north       │ │
│  ├──────────────────────┤             └──────────────────────────────┘ │
│  │ Worker:              │                          │                   │
│  │ observer-3bay_north  │                          ▼ Trigger           │
│  ├──────────────────────┤             ┌──────────────────────────────┐ │
│  │ Tasks (all on        │             │       Tracking               │ │
│  │  Observer):          │             │       3bay_north             │ │
│  │ 1. capture_images    │             ├──────────────────────────────┤ │
│  │ 2. check_should_     │             │ queue: proc_3bay_north       │ │
│  │    proceed           │             ├──────────────────────────────┤ │
│  │ 3. run_inference     │             │ Worker:                      │ │
│  │ 4. trigger_fusion    │             │ processor-3bay_north         │ │
│  └──────────────────────┘             └──────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Bay 설정 및 DAG Factory

### 4.1 통합 Bay 설정 구조

```python
# config/bay_configs.py
"""
Bay별 통합 설정
- 새로운 Bay 추가 시 여기만 수정
- Queue 이름은 bay_id에서 자동 생성
"""

from typing import Dict, Any, List

BAY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "3bay_north": {
        # === 기본 정보 ===
        "description": "3Bay North Zone",
        "enabled": True,

        # === 네트워크 설정 ===
        "network": {
            "camera_subnet": "10.150.160.0/24",
            "observer_ip": "10.150.160.1",
        },

        # === 스케줄링 설정 ===
        "schedule": {
            "observation": "*/10 * * * *",      # 10분마다
            "fusion": None,                      # Trigger 기반
            "tracking": None,                    # Trigger 기반
        },
        "operating_hours": [
            (8, 30, 9, 20),
            (11, 0, 11, 50),
            (13, 30, 14, 20),
            (16, 0, 16, 50),
        ],
        "skip_times": [
            (8, 40, 3),
            (11, 10, 3),
            (13, 40, 3),
            (16, 10, 3),
        ],

        # === 카메라 설정 ===
        "cameras": [
            {"name": "C_2", "host": "10.150.160.183", "port": 80},
            {"name": "C_4", "host": "10.150.160.184", "port": 80},
            {"name": "C_6", "host": "10.150.160.193", "port": 80},
            {"name": "C_7", "host": "10.150.160.194", "port": 80},
            {"name": "C_9", "host": "10.150.160.209", "port": 80},
            {"name": "C_11", "host": "10.150.160.211", "port": 80},
            {"name": "D_2", "host": "10.150.160.221", "port": 80},
            {"name": "D_4", "host": "10.150.160.222", "port": 80},
            {"name": "D_6", "host": "10.150.160.224", "port": 80},
            {"name": "D_7", "host": "10.150.160.225", "port": 80},
            {"name": "D_9", "host": "10.150.160.226", "port": 80},
            {"name": "D_11", "host": "10.150.160.227", "port": 80},
        ],

        # === 이미지 획득 설정 ===
        "image_provider": "onvif",  # "onvif" | "file"
        "image_provider_config": {
            "timeout": 10,
        },

        # === 모델 설정 (Bay별 다른 모델 사용 가능) ===
        "models": {
            "yolo_detection": "checkpoints/objectdetection_yolo/best_250408.pt",
            "sam_segmentation": "checkpoints/segmentation_sam/sam_vit_h_4b8939.pth",
            "classification": "checkpoints/stage_cls/best_250312.pt",
        },

        # === Factory Layout (Bay별 다른 레이아웃) ===
        "factory_layout": {
            "width": 150,
            "height": 40,
            "areas": {
                "A25": {"top_left": (2, 0), "bottom_right": (18, 20)},
                # ... 기타 영역
            },
        },
    },

    "4bay": {
        "description": "4Bay Zone",
        "enabled": False,  # 준비되면 True로 변경

        "network": {
            "camera_subnet": "10.150.161.0/24",
            "observer_ip": "10.150.161.1",
        },

        "schedule": {
            "observation": "*/10 * * * *",
            "fusion": None,
            "tracking": None,
        },
        "operating_hours": [
            (8, 0, 12, 0),
            (13, 0, 17, 0),
        ],
        "skip_times": [],

        "cameras": [
            # 4bay 카메라 설정
        ],

        "image_provider": "onvif",
        "image_provider_config": {},
        "models": {},
        "factory_layout": {},
    },

    "5bay": {
        "description": "5Bay Zone",
        "enabled": False,

        "network": {
            "camera_subnet": "10.150.162.0/24",
            "observer_ip": "10.150.162.1",
        },

        "schedule": {
            "observation": "*/10 * * * *",
            "fusion": None,
            "tracking": None,
        },
        "operating_hours": [],
        "skip_times": [],
        "cameras": [],
        "image_provider": "onvif",
        "image_provider_config": {},
        "models": {},
        "factory_layout": {},
    },
}


def get_observer_queue(bay_id: str) -> str:
    """Observer Queue 이름 생성"""
    return f"obs_{bay_id}"


def get_processor_queue(bay_id: str) -> str:
    """Processor Queue 이름 생성"""
    return f"proc_{bay_id}"


def get_enabled_bays() -> Dict[str, Dict[str, Any]]:
    """활성화된 Bay만 반환"""
    return {
        bay_id: config
        for bay_id, config in BAY_CONFIGS.items()
        if config.get("enabled", False)
    }
```

### 4.2 통합 DAG Factory (Observation + Fusion + Tracking)

```python
# dags/bay_dag_factory.py
"""
Bay별 전체 파이프라인 DAG Factory

각 Bay에 대해 다음 DAG들을 동적으로 생성:
1. observation_{bay_id}_dag  → Observer에서 실행 (Bay 물리 서버)
2. fusion_{bay_id}_dag       → Processor에서 실행 (Main Server Host)
3. tracking_{bay_id}_dag     → Processor에서 실행 (Main Server Host)
"""

from airflow import DAG
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from typing import Dict, Any
import sys

sys.path.insert(0, '/opt/airflow')
from config.bay_configs import (
    BAY_CONFIGS,
    get_observer_queue,
    get_processor_queue,
    get_enabled_bays,
)


# ============================================================================
# 공통 헬퍼 함수
# ============================================================================

def is_in_operating_hours(dt: datetime, operating_hours: list) -> bool:
    """운영 시간 내인지 확인"""
    current_minutes = dt.hour * 60 + dt.minute
    for start_h, start_m, end_h, end_m in operating_hours:
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= current_minutes <= end_minutes:
            return True
    return False


def should_skip_execution(dt: datetime, skip_times: list) -> bool:
    """스킵 시간인지 확인"""
    current_minutes = dt.hour * 60 + dt.minute
    for skip_h, skip_m, tolerance in skip_times:
        skip_minutes = skip_h * 60 + skip_m
        if abs(current_minutes - skip_minutes) <= tolerance:
            return True
    return False


# Note: check_should_execute 함수는 더 이상 사용되지 않음
# 대신 check_execution_conditions 함수가 Observer에서 실행되어
# 캡처된 이미지와 함께 조건을 확인 (아래 Task 구현 함수 참조)


# ============================================================================
# Observation DAG Factory
# ============================================================================

def create_observation_dag(bay_id: str, config: Dict[str, Any]) -> DAG:
    """
    Observation DAG 생성 - 카메라 캡처 및 AI 추론

    실행 위치: Bay별 물리 서버의 Observer
    """

    dag_id = f"observation_{bay_id}_dag"
    observer_queue = get_observer_queue(bay_id)

    default_args = {
        'owner': 'samho-ops',
        'depends_on_past': False,
        'email_on_failure': True,
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    }

    dag = DAG(
        dag_id=dag_id,
        description=f'{config["description"]} - CCTV Observation',
        default_args=default_args,
        schedule=config["schedule"]["observation"],
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['observation', bay_id, 'cctv', 'observer'],
    )

    with dag:
        # ========================================================================
        # 모든 Task는 Bay 물리 서버의 Observer에서 실행
        # Main Server(Scheduler)는 스케줄에 따라 Task를 Observer Queue에 dispatch만 수행
        # ========================================================================

        # Task 1: 이미지 캡처 (Bay 물리 서버의 Observer에서 실행)
        capture_images = PythonOperator(
            task_id='capture_images',
            python_callable=run_capture_task,
            op_kwargs={
                "bay_id": bay_id,
                "cameras": config["cameras"],
                "image_provider_type": config["image_provider"],
                "image_provider_config": config.get("image_provider_config", {}),
            },
            queue=observer_queue,  # Observer 전용 큐 (Bay 물리 서버)
        )

        # Task 2: 수행 여부 판단 (Bay 물리 서버의 Observer에서 실행)
        # - 운영 시간, 이미지 조도, 기타 조건 체크
        # - 조건 불충족 시 downstream task 스킵
        check_should_proceed = ShortCircuitOperator(
            task_id='check_should_proceed',
            python_callable=check_execution_conditions,
            op_kwargs={
                "bay_id": bay_id,
                "operating_hours": config["operating_hours"],
                "skip_times": config["skip_times"],
            },
            queue=observer_queue,  # Observer 전용 큐 (Bay 물리 서버)
        )

        # Task 3: AI 추론 및 결과 저장 (Bay 물리 서버의 Observer에서 실행)
        run_inference = PythonOperator(
            task_id='run_inference',
            python_callable=run_inference_task,
            op_kwargs={
                "bay_id": bay_id,
            },
            queue=observer_queue,  # Observer 전용 큐 (Bay 물리 서버)
        )

        # Task 4: Fusion DAG 트리거 (Bay 물리 서버의 Observer에서 실행)
        trigger_fusion = TriggerDagRunOperator(
            task_id='trigger_fusion',
            trigger_dag_id=f'fusion_{bay_id}_dag',
            conf={
                'batch_id': '{{ ti.xcom_pull(task_ids="run_inference")["batch_id"] }}',
                'bay_id': bay_id,
                'observation_result_path': '{{ ti.xcom_pull(task_ids="run_inference")["result_path"] }}',
            },
            wait_for_completion=False,
            queue=observer_queue,  # Observer 전용 큐 (Bay 물리 서버)
        )

        capture_images >> check_should_proceed >> run_inference >> trigger_fusion

    return dag


# ============================================================================
# Fusion DAG Factory
# ============================================================================

def create_fusion_dag(bay_id: str, config: Dict[str, Any]) -> DAG:
    """
    Fusion DAG 생성 - 다중 카메라 융합 분석

    실행 위치: Main Server Host의 Processor
    """

    dag_id = f"fusion_{bay_id}_dag"
    processor_queue = get_processor_queue(bay_id)

    default_args = {
        'owner': 'samho-ops',
        'depends_on_past': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    }

    dag = DAG(
        dag_id=dag_id,
        description=f'{config["description"]} - Multi-Camera Fusion',
        default_args=default_args,
        schedule=None,  # Trigger 기반
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['fusion', bay_id, 'processor'],
    )

    with dag:
        # Main Server Host의 Processor에서 실행
        process_fusion = PythonOperator(
            task_id='process_fusion',
            python_callable=run_fusion_task,
            op_kwargs={"bay_id": bay_id},
            queue=processor_queue,  # Processor 전용 큐 (Main Server Host)
        )

        trigger_tracking = TriggerDagRunOperator(
            task_id='trigger_tracking',
            trigger_dag_id=f'tracking_{bay_id}_dag',
            conf={
                'batch_id': '{{ dag_run.conf["batch_id"] }}',
                'bay_id': bay_id,
                'fusion_result_path': '{{ ti.xcom_pull(task_ids="process_fusion")["result_path"] }}',
            },
            wait_for_completion=False,
        )

        process_fusion >> trigger_tracking

    return dag


# ============================================================================
# Tracking DAG Factory
# ============================================================================

def create_tracking_dag(bay_id: str, config: Dict[str, Any]) -> DAG:
    """
    Tracking DAG 생성 - 블록 이동 추적 및 기록

    실행 위치: Main Server Host의 Processor
    """

    dag_id = f"tracking_{bay_id}_dag"
    processor_queue = get_processor_queue(bay_id)

    default_args = {
        'owner': 'samho-ops',
        'depends_on_past': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    }

    dag = DAG(
        dag_id=dag_id,
        description=f'{config["description"]} - Tracking',
        default_args=default_args,
        schedule=None,  # Trigger 기반
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=['tracking', bay_id, 'processor'],
    )

    with dag:
        # Main Server Host의 Processor에서 실행
        process_tracking = PythonOperator(
            task_id='process_tracking',
            python_callable=run_tracking_task,
            op_kwargs={"bay_id": bay_id},
            queue=processor_queue,  # Processor 전용 큐 (Main Server Host)
        )

    return dag


# ============================================================================
# Task 구현 함수 (Placeholder)
# ============================================================================

def run_capture_task(bay_id: str, cameras: list, image_provider_type: str,
                     image_provider_config: dict, **context) -> dict:
    """
    이미지 캡처 태스크 실행

    실행 위치: Bay 물리 서버의 Observer
    """
    from src.observation.image_provider import ImageProviderFactory

    logical_date = context['logical_date']
    batch_id = f"{bay_id}_{logical_date.strftime('%Y%m%d-%H%M%S')}"

    try:
        # 1. Image Provider 생성
        image_provider = ImageProviderFactory.create(
            image_provider_type,
            **image_provider_config,
        )

        # 2. 이미지 획득 (카메라 접근 - Observer에서만 가능)
        captured_images = image_provider.get_images(
            reference_time=logical_date,
            cameras=cameras,
            bay_id=bay_id,
        )

        # 3. 캡처 결과를 XCom에 저장 (다음 Task에서 사용)
        return {
            'success': True,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'captured_images': captured_images,  # XCom으로 전달
            'captured_count': len(captured_images),
        }

    except Exception as e:
        return {
            'success': False,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'error_message': str(e),
        }


def check_execution_conditions(bay_id: str, operating_hours: list,
                               skip_times: list, **context) -> bool:
    """
    수행 여부 판단 태스크

    실행 위치: Bay 물리 서버의 Observer

    판단 기준:
    - 운영 시간 내인지 확인
    - 스킵 시간대인지 확인
    - 캡처된 이미지의 조도 확인 (향후 확장)
    - 기타 조건 확인 (향후 확장)

    Returns:
        bool: True면 다음 Task 실행, False면 downstream task 스킵
    """
    logical_date = context['logical_date']

    # 1. 운영 시간 체크
    if not is_in_operating_hours(logical_date, operating_hours):
        print(f"[{bay_id}] Not in operating hours: {logical_date}")
        return False

    # 2. 스킵 시간 체크
    if should_skip_execution(logical_date, skip_times):
        print(f"[{bay_id}] In skip time window: {logical_date}")
        return False

    # 3. 이미지 조도 체크 (향후 확장)
    # captured_images = context['ti'].xcom_pull(task_ids='capture_images')['captured_images']
    # if not check_image_brightness(captured_images):
    #     print(f"[{bay_id}] Image brightness too low")
    #     return False

    # 4. 기타 조건 체크 (향후 확장)
    # ...

    return True


def run_inference_task(bay_id: str, **context) -> dict:
    """
    AI 추론 및 결과 저장 태스크

    실행 위치: Bay 물리 서버의 Observer
    """
    from src.observation.inference_service import InferenceService
    from src.observation.result_saver import ResultSaver

    # 캡처 Task에서 전달받은 데이터
    capture_result = context['ti'].xcom_pull(task_ids='capture_images')
    batch_id = capture_result['batch_id']
    captured_images = capture_result['captured_images']

    try:
        # 1. AI 추론
        inference_service = InferenceService()
        inference_results = inference_service.run_inference(captured_images)

        # 2. 결과 저장 (공유 스토리지에 Pickle 저장)
        result_saver = ResultSaver()
        result = result_saver.save_results(
            captured_images=captured_images,
            inference_results=inference_results,
            batch_id=batch_id,
            bay_id=bay_id,
        )

        return {
            'success': result.success,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'result_path': result.result_path,
            'processed_count': result.processed_count,
        }

    except Exception as e:
        return {
            'success': False,
            'batch_id': batch_id,
            'bay_id': bay_id,
            'error_message': str(e),
        }


def run_fusion_task(bay_id: str, **context) -> dict:
    """
    Fusion 태스크 실행

    실행 위치: Main Server Host의 Processor
    """
    # TODO: 실제 구현
    dag_run_conf = context['dag_run'].conf
    batch_id = dag_run_conf.get('batch_id')

    return {
        'success': True,
        'batch_id': batch_id,
        'bay_id': bay_id,
        'result_path': f'/opt/airflow/data/fusion/{bay_id}/{batch_id}/result.pkl',
    }


def run_tracking_task(bay_id: str, **context) -> dict:
    """
    Tracking 태스크 실행

    실행 위치: Main Server Host의 Processor
    """
    # TODO: 실제 구현
    dag_run_conf = context['dag_run'].conf
    batch_id = dag_run_conf.get('batch_id')

    return {
        'success': True,
        'batch_id': batch_id,
        'bay_id': bay_id,
    }


# ============================================================================
# DAG 동적 생성
# ============================================================================

for bay_id, config in get_enabled_bays().items():
    # 각 Bay에 대해 3개의 DAG 생성
    globals()[f"observation_{bay_id}_dag"] = create_observation_dag(bay_id, config)
    globals()[f"fusion_{bay_id}_dag"] = create_fusion_dag(bay_id, config)
    globals()[f"tracking_{bay_id}_dag"] = create_tracking_dag(bay_id, config)
```

---

## 5. Worker 환경 설정

### 5.1 Observer 환경 변수 (Bay 물리 서버)

각 Bay의 Observer는 해당 Bay의 Observer Queue만 처리:

```ini
# env/prod/observer_3bay_north.env
WORKER_NAME=observer-3bay_north
WORKER_QUEUES=obs_3bay_north
WORKER_CONCURRENCY=4
WORKER_AUTOSCALE=8,4

# 네트워크 설정 (해당 Bay 카메라 접근용)
CAMERA_SUBNET=10.150.160.0/24

# Main Server 연결
MAIN_SERVER_IP=10.150.150.1
POSTGRES_HOST=10.150.150.1
REDIS_HOST=10.150.150.1
POSTGRES_PORT=55432
REDIS_PORT=56379

# 공유 스토리지 (NFS 파일 서버 마운트)
# 운영환경: 별도 NFS 파일 서버 → /mnt/airflow-data 로 마운트
SHARED_DATA_PATH=/mnt/airflow-data
```

```ini
# env/prod/observer_4bay.env
WORKER_NAME=observer-4bay
WORKER_QUEUES=obs_4bay
WORKER_CONCURRENCY=4
WORKER_AUTOSCALE=8,4

CAMERA_SUBNET=10.150.161.0/24

MAIN_SERVER_IP=10.150.150.1
POSTGRES_HOST=10.150.150.1
REDIS_HOST=10.150.150.1
POSTGRES_PORT=55432
REDIS_PORT=56379

# 공유 스토리지 (NFS 파일 서버 마운트)
SHARED_DATA_PATH=/mnt/airflow-data
```

### 5.2 Processor 환경 변수 (Main Server Host)

각 Bay의 Processor는 해당 Bay의 Processor Queue만 처리:

```ini
# env/prod/processor_3bay_north.env
WORKER_NAME=processor-3bay_north
WORKER_QUEUES=proc_3bay_north
WORKER_CONCURRENCY=8
WORKER_AUTOSCALE=16,8

# Main Server와 동일 호스트이므로 localhost 사용 가능
POSTGRES_HOST=localhost
REDIS_HOST=localhost
POSTGRES_PORT=5432
REDIS_PORT=6379

# 공유 스토리지 (NFS 파일 서버 마운트)
# 운영환경: 별도 NFS 파일 서버 → /mnt/airflow-data 로 마운트
SHARED_DATA_PATH=/mnt/airflow-data
```

```ini
# env/prod/processor_4bay.env
WORKER_NAME=processor-4bay
WORKER_QUEUES=proc_4bay
WORKER_CONCURRENCY=8
WORKER_AUTOSCALE=16,8

POSTGRES_HOST=localhost
REDIS_HOST=localhost
POSTGRES_PORT=5432
REDIS_PORT=6379

# 공유 스토리지 (NFS 파일 서버 마운트)
SHARED_DATA_PATH=/mnt/airflow-data
```

### 5.3 개발 환경 Worker 설정 (단일 서버)

개발 환경에서는 단일 서버에서 모든 Worker를 시뮬레이션합니다.
프로젝트의 `./data` 폴더를 공유 스토리지로 사용합니다.

```ini
# env/dev/observer_3bay_north.env
WORKER_NAME=observer-dev-3bay_north
WORKER_QUEUES=obs_3bay_north,default

# 개발환경: 로컬 ./data 폴더 사용 (Docker volume mount)
# 컨테이너 내부 경로는 /opt/airflow/data
```

```ini
# env/dev/processor_3bay_north.env
WORKER_NAME=processor-dev-3bay_north
WORKER_QUEUES=proc_3bay_north

# 개발환경: 로컬 ./data 폴더 사용 (Docker volume mount)
# 컨테이너 내부 경로는 /opt/airflow/data
```

### 5.4 Docker Compose 설정

#### 개발 환경 (docker-compose.dev.yaml)

```yaml
# docker-compose.dev.yaml (개발환경 - 단일 서버)
x-airflow-common: &airflow-common
  image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:3.0.0}
  volumes:
    - ./data:/opt/airflow/data          # 공유 스토리지 (개발환경: 로컬 ./data)
    - ./dags:/opt/airflow/dags
    - ./config:/opt/airflow/config
    - ./src:/opt/airflow/src

services:
  airflow-observer-dev:
    <<: *airflow-common
    command: celery worker --queues ${WORKER_QUEUES}
    hostname: ${WORKER_NAME}
    # ...

  airflow-processor-dev:
    <<: *airflow-common
    command: celery worker --queues ${WORKER_QUEUES}
    hostname: ${WORKER_NAME}
    # ...
```

#### 운영 환경 (docker-compose.prod.yaml)

```yaml
# docker-compose_observer.yaml (Bay 물리 서버용 - 운영환경)
services:
  airflow-observer:
    image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:3.0.0}
    command: celery worker --queues ${WORKER_QUEUES}
    hostname: ${WORKER_NAME}
    environment:
      AIRFLOW__CELERY__WORKER_CONCURRENCY: ${WORKER_CONCURRENCY:-4}
      AIRFLOW__CELERY__WORKER_AUTOSCALE: ${WORKER_AUTOSCALE:-8,4}
    volumes:
      # NFS 파일 서버가 /mnt/airflow-data 에 마운트되어 있어야 함
      - /mnt/airflow-data:/opt/airflow/data
    # ...
```

```yaml
# docker-compose_processor.yaml (Main Server Host용 - 운영환경)
services:
  airflow-processor:
    image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:3.0.0}
    command: celery worker --queues ${WORKER_QUEUES}
    hostname: ${WORKER_NAME}
    environment:
      AIRFLOW__CELERY__WORKER_CONCURRENCY: ${WORKER_CONCURRENCY:-8}
      AIRFLOW__CELERY__WORKER_AUTOSCALE: ${WORKER_AUTOSCALE:-16,8}
    volumes:
      # NFS 파일 서버가 /mnt/airflow-data 에 마운트되어 있어야 함
      - /mnt/airflow-data:/opt/airflow/data
    # ...
```

### 5.5 Worker 시작 스크립트

```bash
#!/bin/bash
# start-observers.sh - Bay별 Observer 시작 (Bay 물리 서버에서 실행)

ENVIRONMENT=${1:-dev}
BAY_ID=${2:-3bay_north}

echo "Starting observer for $BAY_ID..."

ENV_FILE="env/${ENVIRONMENT}/observer_${BAY_ID}.env"

if [ -f "$ENV_FILE" ]; then
    docker compose \
        --file docker-compose_observer.yaml \
        --env-file env/base.env \
        --env-file "$ENV_FILE" \
        --project-name "airflow-observer-${BAY_ID}" \
        up -d
else
    echo "Error: $ENV_FILE not found"
    exit 1
fi
```

```bash
#!/bin/bash
# start-processors.sh - Bay별 Processor 시작 (Main Server Host에서 실행)

ENVIRONMENT=${1:-dev}
BAYS=${2:-"3bay_north 4bay 5bay"}

for BAY in $BAYS; do
    echo "Starting processor for $BAY..."

    ENV_FILE="env/${ENVIRONMENT}/processor_${BAY}.env"

    if [ -f "$ENV_FILE" ]; then
        docker compose \
            --file docker-compose_processor.yaml \
            --env-file env/base.env \
            --env-file "$ENV_FILE" \
            --project-name "airflow-processor-${BAY}" \
            up -d
    else
        echo "Warning: $ENV_FILE not found, skipping $BAY"
    fi
done
```

---

## 6. Image Provider - Strategy Pattern

### 6.1 설계 원칙

이미지 획득 방식을 **Strategy Pattern**으로 구현하여:
- 실시간 캡처 (ONVIF)
- 파일 기반 로드 (백테스트)
- 향후 추가될 다른 소스 (예: RTSP 스트림, HTTP API 등)

를 동일한 인터페이스로 처리합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    ImageProvider (ABC)                       │
│  ─────────────────────────────────────────────────────────  │
│  + get_images(reference_time, cameras) -> List[CapturedImage]│
└─────────────────────────────────────────────────────────────┘
                              △
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ OnvifImageProvider│ │ FileImageProvider │ │ (Future: RTSP)  │
│                  │ │                  │ │                  │
│ - 실시간 캡처    │ │ - 저장된 파일 로드│ │ - 스트림 캡처    │
│ - ONVIF 프로토콜 │ │ - 백테스트용     │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### 6.2 구현 코드

```python
# src/observation/image_provider/__init__.py
from .base import ImageProvider
from .onvif_provider import OnvifImageProvider
from .file_provider import FileImageProvider
from .factory import ImageProviderFactory

__all__ = [
    "ImageProvider",
    "OnvifImageProvider",
    "FileImageProvider",
    "ImageProviderFactory",
]
```

```python
# src/observation/image_provider/base.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class CapturedImage:
    """캡처된 이미지 데이터 모델"""
    image_id: str
    camera_name: str
    bay_id: str
    image_data: np.ndarray
    captured_at: datetime


class ImageProvider(ABC):
    """
    이미지 획득을 위한 추상 베이스 클래스 (Strategy Pattern)

    모든 이미지 제공자는 이 인터페이스를 구현해야 합니다.
    """

    @abstractmethod
    def get_images(
        self,
        reference_time: datetime,
        cameras: List[Dict[str, Any]],
        bay_id: str,
    ) -> List[CapturedImage]:
        """
        지정된 시간과 카메라 설정에 따라 이미지를 획득합니다.

        Args:
            reference_time: 기준 시간 (캡처 시간 또는 백테스트 기준 시간)
            cameras: 카메라 설정 리스트
            bay_id: Bay 식별자

        Returns:
            List[CapturedImage]: 획득된 이미지 리스트
        """
        pass

    @abstractmethod
    def validate_config(self, cameras: List[Dict[str, Any]]) -> bool:
        """
        카메라 설정의 유효성을 검사합니다.

        Args:
            cameras: 카메라 설정 리스트

        Returns:
            bool: 유효성 검사 통과 여부
        """
        pass
```

```python
# src/observation/image_provider/onvif_provider.py
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

from .base import ImageProvider, CapturedImage


class OnvifImageProvider(ImageProvider):
    """
    ONVIF 프로토콜을 사용하여 카메라에서 실시간으로 이미지를 캡처합니다.

    기존 legacy_reference/cctv_event_detector/repository/onvif_repository.py의
    로직을 래핑합니다.
    """

    def __init__(self, timeout: int = 10):
        """
        Args:
            timeout: 카메라 연결 타임아웃 (초)
        """
        self.timeout = timeout

    def get_images(
        self,
        reference_time: datetime,
        cameras: List[Dict[str, Any]],
        bay_id: str,
    ) -> List[CapturedImage]:
        """모든 카메라에서 실시간으로 이미지를 캡처합니다."""
        captured_images = []

        for camera_config in cameras:
            try:
                image = self._capture_single_camera(
                    camera_config, reference_time, bay_id
                )
                if image:
                    captured_images.append(image)
            except Exception as e:
                print(f"Camera {camera_config['name']} capture failed: {e}")
                continue

        return captured_images

    def _capture_single_camera(
        self,
        config: Dict[str, Any],
        reference_time: datetime,
        bay_id: str,
    ) -> CapturedImage:
        """단일 카메라에서 이미지를 캡처합니다."""
        camera_name = config["name"]
        image_id = f"{bay_id}_{camera_name}_{reference_time.strftime('%Y%m%d%H%M%S')}"

        image_data = self._get_snapshot(config)

        return CapturedImage(
            image_id=image_id,
            camera_name=camera_name,
            bay_id=bay_id,
            image_data=image_data,
            captured_at=reference_time,
        )

    def _get_snapshot(self, config: Dict[str, Any]) -> np.ndarray:
        """ONVIF 스냅샷 획득"""
        # TODO: 실제 ONVIF 구현
        pass

    def validate_config(self, cameras: List[Dict[str, Any]]) -> bool:
        required_fields = ["name", "host", "port"]
        for camera in cameras:
            for field in required_fields:
                if field not in camera:
                    return False
        return True
```

```python
# src/observation/image_provider/file_provider.py
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import re

from .base import ImageProvider, CapturedImage


class FileImageProvider(ImageProvider):
    """
    저장된 이미지 파일에서 이미지를 로드합니다.
    백테스트 및 개발/테스트 용도로 사용됩니다.
    """

    TIMESTAMP_PATTERN = re.compile(r'(\d{14})')

    def __init__(self, base_path: str, time_tolerance_hours: float = 1.0):
        self.base_path = Path(base_path)
        self.time_tolerance = timedelta(hours=time_tolerance_hours)

    def get_images(
        self,
        reference_time: datetime,
        cameras: List[Dict[str, Any]],
        bay_id: str,
    ) -> List[CapturedImage]:
        """기준 시간에 맞는 저장된 이미지 파일을 로드합니다."""
        captured_images = []

        for camera_config in cameras:
            camera_name = camera_config["name"]
            try:
                image = self._load_image_for_camera(
                    camera_name, reference_time, bay_id
                )
                if image:
                    captured_images.append(image)
            except Exception as e:
                print(f"Camera {camera_name} file load failed: {e}")
                continue

        return captured_images

    def _load_image_for_camera(
        self,
        camera_name: str,
        reference_time: datetime,
        bay_id: str,
    ) -> Optional[CapturedImage]:
        """특정 카메라의 기준 시간에 가장 가까운 이미지를 로드합니다."""
        camera_dir = self.base_path / camera_name

        if not camera_dir.exists():
            return None

        image_files = list(camera_dir.glob("*.png")) + list(camera_dir.glob("*.jpg"))
        if not image_files:
            return None

        best_file = self._find_closest_image(image_files, reference_time)
        if best_file is None:
            return None

        image_data = cv2.imread(str(best_file))
        if image_data is None:
            return None

        file_timestamp = self._extract_timestamp(best_file.name)
        captured_at = file_timestamp or reference_time

        image_id = f"{bay_id}_{camera_name}_{captured_at.strftime('%Y%m%d%H%M%S')}"

        return CapturedImage(
            image_id=image_id,
            camera_name=camera_name,
            bay_id=bay_id,
            image_data=image_data,
            captured_at=captured_at,
        )

    def _find_closest_image(self, files: List[Path], reference_time: datetime) -> Optional[Path]:
        best_file = None
        best_diff = self.time_tolerance

        for file_path in files:
            timestamp = self._extract_timestamp(file_path.name)
            if timestamp is None:
                continue

            diff = abs(timestamp - reference_time)
            if diff <= self.time_tolerance and diff < best_diff:
                best_diff = diff
                best_file = file_path

        return best_file

    def _extract_timestamp(self, filename: str) -> Optional[datetime]:
        match = self.TIMESTAMP_PATTERN.search(filename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
            except ValueError:
                return None
        return None

    def validate_config(self, cameras: List[Dict[str, Any]]) -> bool:
        for camera in cameras:
            if "name" not in camera:
                return False
        return True
```

```python
# src/observation/image_provider/factory.py
from typing import Dict, Any

from .base import ImageProvider
from .onvif_provider import OnvifImageProvider
from .file_provider import FileImageProvider


class ImageProviderFactory:
    """Image Provider Factory (Strategy Pattern Context)"""

    _providers: Dict[str, type] = {
        "onvif": OnvifImageProvider,
        "file": FileImageProvider,
    }

    @classmethod
    def create(cls, provider_type: str, **kwargs: Any) -> ImageProvider:
        provider_class = cls._providers.get(provider_type)
        if provider_class is None:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider: {provider_type}. Available: {available}")
        return provider_class(**kwargs)

    @classmethod
    def register(cls, name: str, provider_class: type) -> None:
        if not issubclass(provider_class, ImageProvider):
            raise TypeError(f"{provider_class.__name__} must inherit from ImageProvider")
        cls._providers[name] = provider_class
```

---

## 7. 데이터 흐름 및 저장소

### 7.1 디렉토리 구조

```
airflow-test-01/
├── dags/
│   └── bay_dag_factory.py           # 통합 DAG Factory
│
├── config/
│   └── bay_configs.py               # Bay별 설정
│
├── src/
│   ├── observation/
│   │   ├── image_provider/
│   │   │   ├── base.py
│   │   │   ├── onvif_provider.py
│   │   │   ├── file_provider.py
│   │   │   └── factory.py
│   │   ├── inference_service.py
│   │   └── result_saver.py
│   │
│   ├── fusion/                      # Fusion 모듈
│   │   └── ...
│   │
│   └── tracking/                    # Tracking 모듈
│       └── ...
│
├── data/                            # 공유 스토리지
│   │                                # - 개발: 로컬 ./data 폴더 (Docker volume mount)
│   │                                # - 운영: 별도 NFS 파일 서버 마운트
│   ├── observation/                 # Observation 결과 저장
│   │   └── {bay_id}/
│   │       └── {batch_id}/
│   │           └── result.pkl       # 캡처 이미지 + 추론 결과
│   │
│   ├── fusion/                      # Fusion 결과 저장
│   │   └── {bay_id}/
│   │       └── {batch_id}/
│   │           └── result.pkl
│   │
│   └── tracking/                    # Tracking 결과 저장
│       └── {bay_id}/
│           └── {batch_id}/
│               └── result.pkl
│
├── env/
│   ├── base.env
│   ├── dev/
│   │   ├── observer_3bay_north.env
│   │   ├── observer_4bay.env
│   │   ├── processor_3bay_north.env
│   │   └── processor_4bay.env
│   └── prod/
│       ├── observer_3bay_north.env
│       ├── observer_4bay.env
│       ├── processor_3bay_north.env
│       └── processor_4bay.env
│
└── legacy_reference/
```

### 7.2 XCom 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        XCom Data Flow                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Observation DAG (Bay 물리 서버의 Observer에서 실행)                    │
│  ├─ capture_images task                                                 │
│  │    └─ XCom Push: { batch_id, captured_images, ... }                 │
│  │                                                                      │
│  ├─ check_should_proceed task (ShortCircuit)                           │
│  │    └─ 조건 불충족 시 downstream skip                                 │
│  │                                                                      │
│  ├─ run_inference task                                                  │
│  │    └─ XCom Push:                                                    │
│  │        {                                                            │
│  │            "success": true,                                         │
│  │            "batch_id": "3bay_north_20250702-161000",                │
│  │            "bay_id": "3bay_north",                                  │
│  │            "result_path": "/data/observation/3bay.../result.pkl",   │
│  │            "processed_count": 12                                    │
│  │        }                                                            │
│  │                                                                      │
│  └─ trigger_fusion task                                                 │
│                         │                                               │
│                         ▼ TriggerDagRunOperator (conf)                  │
│                                                                         │
│  Fusion DAG (Main Server Host의 Processor에서 실행)                     │
│  └─ process_fusion task                                                 │
│      ├─ dag_run.conf로 Observation 결과 수신                            │
│      ├─ 공유 스토리지에서 Pickle 로드                                   │
│      └─ XCom Push:                                                      │
│          {                                                              │
│              "success": true,                                           │
│              "batch_id": "3bay_north_20250702-161000",                  │
│              "bay_id": "3bay_north",                                    │
│              "result_path": "/data/fusion/3bay.../result.pkl"           │
│          }                                                              │
│                         │                                               │
│                         ▼ TriggerDagRunOperator (conf)                  │
│                                                                         │
│  Tracking DAG (Main Server Host의 Processor에서 실행)                   │
│  └─ process_tracking task                                               │
│      └─ dag_run.conf로 Fusion 결과 수신                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 공유 스토리지 구성

Observer와 Processor 간 데이터 공유를 위한 스토리지 구성:

#### 7.3.1 개발 환경: 로컬 ./data 폴더 사용

개발 환경에서는 프로젝트의 `./data` 폴더를 공유 스토리지로 사용합니다.
모든 컨테이너(Main, Observer, Processor)가 동일한 호스트에서 실행되므로,
Docker volume mount를 통해 동일한 디렉토리를 공유합니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Development Environment - Local ./data Folder               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Host Machine (Development)                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Project Directory: ./data/                                       │   │
│  │ ├── observation/{bay_id}/{batch_id}/result.pkl                  │   │
│  │ ├── fusion/{bay_id}/{batch_id}/result.pkl                       │   │
│  │ └── tracking/{bay_id}/{batch_id}/result.pkl                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼ Docker Volume Mount                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ All Containers: ./data → /opt/airflow/data                       │   │
│  │  - airflow-main                                                  │   │
│  │  - airflow-observer-dev                                          │   │
│  │  - airflow-processor-dev                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**docker-compose.dev.yaml 설정 예시:**
```yaml
x-airflow-common: &airflow-common
  volumes:
    - ./data:/opt/airflow/data  # 공유 스토리지 (개발환경)
    - ./dags:/opt/airflow/dags
    - ./config:/opt/airflow/config
    - ./src:/opt/airflow/src
```

#### 7.3.2 운영 환경: 별도 파일 서버 (NFS)

운영 환경에서는 별도의 파일 서버를 NFS 서버로 사용합니다.
Main Server, 모든 Observer, 모든 Processor가 이 파일 서버에 NFS 마운트합니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│             Production Environment - Dedicated File Server               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dedicated NFS File Server (예: 10.150.150.10)                    │   │
│  │                                                                   │   │
│  │ Export Path: /exports/airflow-data                               │   │
│  │ ├── observation/{bay_id}/{batch_id}/result.pkl                   │   │
│  │ ├── fusion/{bay_id}/{batch_id}/result.pkl                        │   │
│  │ └── tracking/{bay_id}/{batch_id}/result.pkl                      │   │
│  │                                                                   │   │
│  │ NFS Export 설정:                                                  │   │
│  │ /exports/airflow-data  10.150.150.0/24(rw,sync,no_subtree_check) │   │
│  │ /exports/airflow-data  10.150.160.0/24(rw,sync,no_subtree_check) │   │
│  │ /exports/airflow-data  10.150.161.0/24(rw,sync,no_subtree_check) │   │
│  │ /exports/airflow-data  10.150.162.0/24(rw,sync,no_subtree_check) │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           │ NFS Mount                                                   │
│           │                                                             │
│           ├─── Main Server (10.150.150.1)                              │
│           │    mount -t nfs 10.150.150.10:/exports/airflow-data        │
│           │          → /mnt/airflow-data                               │
│           │                                                             │
│           ├─── Observer 3bay_north (10.150.160.1)                      │
│           │    mount -t nfs 10.150.150.10:/exports/airflow-data        │
│           │          → /mnt/airflow-data                               │
│           │                                                             │
│           ├─── Observer 4bay (10.150.161.1)                            │
│           │    mount -t nfs 10.150.150.10:/exports/airflow-data        │
│           │          → /mnt/airflow-data                               │
│           │                                                             │
│           └─── Observer 5bay (10.150.162.1)                            │
│                mount -t nfs 10.150.150.10:/exports/airflow-data        │
│                      → /mnt/airflow-data                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 7.3.3 개발 → 운영 이관 전략

개발에서 운영으로 이관 시, 코드 변경 없이 환경 변수와 마운트 설정만 변경:

| 항목 | 개발 환경 | 운영 환경 |
|------|----------|----------|
| **호스트 경로** | `./data` (프로젝트 폴더) | `/mnt/airflow-data` (NFS 마운트) |
| **컨테이너 경로** | `/opt/airflow/data` | `/opt/airflow/data` (동일) |
| **환경 변수** | `DATA_PATH=./data` | `DATA_PATH=/mnt/airflow-data` |
| **설정 변경** | docker-compose volume | NFS mount + docker-compose volume |

**docker-compose.prod.yaml 설정 예시:**
```yaml
x-airflow-common: &airflow-common
  volumes:
    - /mnt/airflow-data:/opt/airflow/data  # NFS 마운트 포인트
    - ./dags:/opt/airflow/dags
    - ./config:/opt/airflow/config
    - ./src:/opt/airflow/src
```

**코드에서의 경로 사용 (변경 없음):**
```python
# 개발/운영 환경 모두 동일한 코드 사용
OBSERVATION_DATA_PATH = "/opt/airflow/data/observation"
FUSION_DATA_PATH = "/opt/airflow/data/fusion"
TRACKING_DATA_PATH = "/opt/airflow/data/tracking"

# 결과 저장 경로
result_path = f"/opt/airflow/data/observation/{bay_id}/{batch_id}/result.pkl"
```

#### 7.3.4 Alternative: S3/MinIO (선택적)

대규모 데이터 또는 클라우드 환경에서는 S3/MinIO를 대안으로 사용 가능:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Alternative: S3/MinIO                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ MinIO Server (On-premise) 또는 AWS S3                           │   │
│  │ Bucket: airflow-data                                            │   │
│  │ - observation/{bay_id}/{batch_id}/                              │   │
│  │ - fusion/{bay_id}/{batch_id}/                                   │   │
│  │ - tracking/{bay_id}/{batch_id}/                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ※ S3 사용 시 코드 수정 필요 (boto3/s3fs 라이브러리 사용)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 마이그레이션 단계별 계획

### Phase 1: 기반 구조 구축

| 순서 | 작업 | 상세 |
|------|------|------|
| 1.1 | 디렉토리 구조 생성 | `config/`, `src/observation/`, `src/fusion/`, `src/tracking/`, `data/` |
| 1.2 | Bay 설정 파일 작성 | `config/bay_configs.py` |
| 1.3 | Worker 환경 변수 설정 | Observer/Processor별 env 파일 생성 |
| 1.4 | 공유 스토리지 설정 | 개발: `./data` 폴더 생성, 운영: NFS 파일 서버 마운트 설정 |

### Phase 2: Observation 모듈 구현

| 순서 | 작업 | 상세 |
|------|------|------|
| 2.1 | ImageProvider ABC 구현 | Strategy Pattern 인터페이스 |
| 2.2 | OnvifImageProvider 구현 | 실시간 캡처 |
| 2.3 | FileImageProvider 구현 | 백테스트용 파일 로드 |
| 2.4 | InferenceService 구현 | AI 추론 로직 |
| 2.5 | ResultSaver 구현 | Pickle 저장 |

### Phase 3: DAG Factory 구현

| 순서 | 작업 | 상세 |
|------|------|------|
| 3.1 | bay_dag_factory.py 작성 | 통합 DAG Factory |
| 3.2 | Observation DAG 검증 | 3bay_north Observer 테스트 |
| 3.3 | Fusion DAG 구현 | Placeholder → 실제 구현 |
| 3.4 | Tracking DAG 구현 | Placeholder → 실제 구현 |

### Phase 4: Worker 배포

| 순서 | 작업 | 상세 |
|------|------|------|
| 4.1 | 3bay_north Observer 배포 | Bay 물리 서버에 Observer 배포 |
| 4.2 | 3bay_north Processor 배포 | Main Server Host에 Processor 배포 |
| 4.3 | 전체 파이프라인 E2E 테스트 | Observation → Fusion → Tracking |
| 4.4 | 추가 Bay Observer/Processor 배포 | 4bay, 5bay 순차 배포 |
| 4.5 | 기존 시스템 비활성화 | producer_main.py 중지 |

---

## 9. Bay 추가 가이드 (Quick Reference)

새로운 Bay를 추가하려면:

### Step 1: Bay 설정 추가

```python
# config/bay_configs.py
BAY_CONFIGS["6bay_south"] = {
    "description": "6Bay South Zone",
    "enabled": True,

    "network": {
        "camera_subnet": "10.150.163.0/24",
        "observer_ip": "10.150.163.1",
    },

    "schedule": {
        "observation": "*/10 * * * *",
        "fusion": None,
        "tracking": None,
    },
    "operating_hours": [(7, 0, 18, 0)],
    "skip_times": [],

    "cameras": [
        {"name": "E_1", "host": "10.150.163.101", "port": 80},
        {"name": "E_2", "host": "10.150.163.102", "port": 80},
        # ...
    ],

    "image_provider": "onvif",
    "image_provider_config": {"timeout": 10},
    "models": {},
    "factory_layout": {},
}
```

### Step 2: Worker 환경 변수 추가

```ini
# env/prod/observer_6bay_south.env
WORKER_NAME=observer-6bay_south
WORKER_QUEUES=obs_6bay_south
CAMERA_SUBNET=10.150.163.0/24
# ...

# env/prod/processor_6bay_south.env
WORKER_NAME=processor-6bay_south
WORKER_QUEUES=proc_6bay_south
# ...
```

### Step 3: Worker 배포

```bash
# Bay 물리 서버에서 Observer 시작
./start-observers.sh prod 6bay_south

# Main Server Host에서 Processor 시작
./start-processors.sh prod "6bay_south"
```

### Step 4: DAG 자동 생성 확인

Airflow UI에서 자동 생성 확인:
- `observation_6bay_south_dag`
- `fusion_6bay_south_dag`
- `tracking_6bay_south_dag`

---

## 10. 주요 변경 사항 요약

| 항목 | AS-IS (기존) | TO-BE (마이그레이션 후) |
|------|-------------|----------------------|
| **용어** | Producer | Observation (Observer) |
| **용어** | Situation Awareness | Fusion |
| **스케줄링** | while 루프 + sleep | Airflow Scheduler + DAG Factory |
| **이미지 획득** | OnvifRepository 직접 사용 | ImageProvider Strategy Pattern |
| **백테스트** | producer_main_backtest.py 별도 | FileImageProvider로 통합 |
| **데이터 저장** | Redis Hash + Pub/Sub | Pickle 파일 (개발: ./data, 운영: NFS 파일 서버) |
| **메타정보 전달** | Redis Publish | XCom + TriggerDagRunOperator |
| **Bay 확장** | 코드 수정 필요 | BAY_CONFIGS 추가만으로 즉시 확장 |
| **Worker 배치** | 단일 Worker | Observer (Bay) + Processor (Central) |
| **네트워크 격리** | 없음 | Observer만 카메라 접근, Processor는 데이터만 |
| **모니터링** | 별도 로깅 | Airflow UI + Flower |

---

## 11. 리스크 및 고려사항

### 11.1 Observer 장애 시

- **문제**: 특정 Bay의 Observer 장애 시 해당 Bay Observation 중단
- **대안**:
  - Bay별 백업 Observer 구성 (Active-Standby)
  - Health check 및 자동 failover 구성

### 11.2 Processor 장애 시

- **문제**: 특정 Bay의 Processor 장애 시 해당 Bay Fusion/Tracking 중단
- **대안**:
  - 동일 Bay를 처리하는 다중 Processor 구성
  - Main Server Host에서 빠른 복구 가능 (컨테이너 재시작)

### 11.3 공유 스토리지 장애

- **문제**: NFS 파일 서버 장애 시 전체 파이프라인 중단
- **대안**:
  - NFS HA 구성 (DRBD + Pacemaker 또는 GlusterFS)
  - S3/MinIO 사용 (고가용성 내장)
  - 각 Observer 로컬 저장 후 비동기 복제
- **개발 환경에서의 완화**:
  - 개발 환경은 로컬 `./data` 폴더 사용으로 NFS 의존성 없음
  - 운영 이관 전 NFS 파일 서버 안정성 충분히 검증 필요

### 11.4 네트워크 장애

- **문제**: Observer ↔ Main Server 네트워크 단절 시
- **대안**:
  - Celery의 내장 재시도 메커니즘 활용
  - Observer 로컬 큐잉 후 재전송
  - 네트워크 모니터링 및 알림 설정

---

## 12. 다음 단계

1. **Fusion 상세 구현**: `run_fusion_task` 실제 로직
2. **Tracking 상세 구현**: `run_tracking_task` 실제 로직
3. **개발 환경 검증**: 로컬 `./data` 폴더 기반 E2E 테스트
4. **운영 환경 스토리지 구성**: 별도 NFS 파일 서버 구축 및 마운트 설정
5. **통합 테스트**: 전체 파이프라인 E2E 테스트 (개발 → 운영 환경)
6. **운영 배포**: Bay별 Observer/Processor 물리 서버 배포
