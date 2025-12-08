### docker compose requirements

- docker compose --file docker-compose_main.yaml --env-file .env.main --profile flower up
- docker compose --project-name airflow-worker-dev --file docker-compose_worker.yaml --env-file .env.worker up -d

### 자동 복구를 위해 autoheal 컨테이너 추가해야함
