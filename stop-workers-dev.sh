#!/bin/bash
# Stop all Airflow Workers for Development Environment

echo "Stopping Airflow Workers..."

# Worker 1
echo "Stopping Worker 1..."
docker compose --project-name airflow-worker-1 \
  --file docker-compose_worker.yaml \
  --env-file .env.worker down

# Worker 2
echo "Stopping Worker 2..."
docker compose --project-name airflow-worker-2 \
  --file docker-compose_worker.yaml \
  --env-file .env.worker2 down

# Add more workers as needed
# Worker 3
# echo "Stopping Worker 3..."
# docker compose --project-name airflow-worker-3 \
#   --file docker-compose_worker.yaml \
#   --env-file .env.worker3 down

echo ""
echo "All workers stopped."
echo "To restart workers: ./start-workers-dev.sh"