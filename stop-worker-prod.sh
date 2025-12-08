#!/bin/bash
# Stop Airflow Worker for Production Environment

echo "Stopping Airflow Worker..."

docker compose --project-name airflow-worker-prod \
  --file docker-compose_worker.yaml \
  --env-file .env.worker down

echo ""
echo "Worker stopped."
echo "To restart worker: ./start-worker-prod.sh"