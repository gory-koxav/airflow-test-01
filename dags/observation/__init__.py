# -*- coding: utf-8 -*-
"""
Observation DAG Module

Observer 워커에서 실행되는 Observation DAG를 정의합니다.

Components:
- tasks.py: TaskFlow API 기반 Task 래퍼 함수
- factory.py: Bay별 Observation DAG 동적 생성

배포 독립성:
- 이 모듈은 Observer 워커에만 배포됩니다.
- Processor 워커에는 processing 모듈만 배포됩니다.
"""
