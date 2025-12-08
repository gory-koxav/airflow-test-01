#!/bin/bash
# Production Environment - Single worker per server
# Run this script on each worker server independently

# Check if env file exists
if [ ! -f ".env.worker" ]; then
    echo "Error: .env.worker file not found!"
    echo "Please ensure .env.worker is configured with production settings."
    exit 1
fi

echo "Starting Airflow Worker for Production Environment..."

# Get worker name from env file
WORKER_NAME=$(grep "^WORKER_NAME=" .env.worker | cut -d'=' -f2)
echo "Starting Worker: ${WORKER_NAME}"

# Start the worker
docker compose --project-name airflow-worker-prod \
  --file docker-compose_worker.yaml \
  --env-file .env.worker up -d

# Check status
echo ""
echo "Checking worker status..."
sleep 10
docker ps --format "table {{.Names}}\t{{.Status}}" | grep airflow-worker

echo ""
echo "Worker started successfully!"
echo "To check logs: docker logs airflow-worker-prod-airflow-worker-1"
echo "To stop worker: ./stop-worker-prod.sh"
echo ""
echo "Important: Ensure the following are properly configured in .env.worker:"
echo "  - MAIN_SERVER_IP (actual IP of main server)"
echo "  - POSTGRES_HOST=\${MAIN_SERVER_IP}"
echo "  - REDIS_HOST=\${MAIN_SERVER_IP}"
echo "  - Unique WORKER_NAME"