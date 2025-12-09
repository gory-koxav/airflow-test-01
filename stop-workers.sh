#!/bin/bash
# Stop all Airflow Workers
# Usage: ./stop-workers.sh [worker_count]

WORKER_COUNT=${1:-10}

echo "Stopping Airflow Workers..."

for i in $(seq 1 $WORKER_COUNT); do
    PROJECT_NAME="airflow-worker-${i}"

    # Check if project exists
    if docker compose --project-name $PROJECT_NAME ps -q 2>/dev/null | grep -q .; then
        echo "Stopping $PROJECT_NAME..."
        docker compose --project-name $PROJECT_NAME --file docker-compose_worker.yaml down
    fi
done

echo ""
echo "All workers stopped."
