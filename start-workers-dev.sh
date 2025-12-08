#!/bin/bash
# Development Environment - Multiple workers on same server
# [PRODUCTION-CHANGE] In production, each worker runs on separate servers

echo "Starting Airflow Workers for Development Environment..."

# Worker 1
echo "Starting Worker 1..."
docker compose --project-name airflow-worker-1 \
  --file docker-compose_worker.yaml \
  --env-file .env.worker up -d

# Worker 2
echo "Starting Worker 2..."
docker compose --project-name airflow-worker-2 \
  --file docker-compose_worker.yaml \
  --env-file .env.worker2 up -d

# Add more workers as needed
# Worker 3
# echo "Starting Worker 3..."
# docker compose --project-name airflow-worker-3 \
#   --file docker-compose_worker.yaml \
#   --env-file .env.worker3 up -d

echo ""
echo "Checking worker status..."
sleep 5
docker ps --format "table {{.Names}}\t{{.Status}}" | grep airflow-worker

echo ""
echo "Workers started successfully!"
echo "To check logs: docker logs <container-name>"
echo "To stop all workers: ./stop-workers-dev.sh"