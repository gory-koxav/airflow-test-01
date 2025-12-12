### 사용 방법
## 도커 이미지 빌드
# 모든 이미지 빌드 (Observer, Processor)
./build-images.sh all

# 개별 빌드
./build-images.sh observer   # ML 이미지 (~10-20분)
./build-images.sh processor  # 경량 이미지 (~2분)

## airflow docker-compose 실행
- chmod +x ./start-main.sh ./start-workers.sh ./stop-workers.sh ./stop-main.sh
# main 시작
- ./start-main.sh dev flower

# 모든 워커 시작
- ./start-workers.sh dev
# Observer만 시작
- ./start-workers.sh dev obs
# Processor만 시작  
- ./start-workers.sh dev proc
# 특정 bay만 시작
- ./start-workers.sh dev obs 64bay
# 모든 워커 중지
- ./stop-workers.sh dev
# Observer만 중지
- ./stop-workers.sh dev obs

# main 중지
- ./stop-main.sh dev flower

### 자동 복구를 위해 autoheal 컨테이너 추가해야함
### log 제거용 DAG 추가해야함