### 사용 방법
- chmod +x ./start-main.sh ./start-workers.sh ./stop-workers.sh ./stop-main.sh
- ./start-main.sh dev flower
- ./start-workers.sh dev 4
- ./stop-workers.sh dev 4
- ./stop-main.sh dev flower

### 자동 복구를 위해 autoheal 컨테이너 추가해야함