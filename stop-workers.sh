#!/bin/bash
# Stop all Airflow Workers
# Usage: ./stop-workers.sh [worker_count]

WORKER_COUNT=${1:-10}

echo "Stopping Airflow Workers..."

for i in $(seq 1 $WORKER_COUNT); do
    PROJECT_NAME="airflow-worker-${i}"
    CONTAINER_NAME="${PROJECT_NAME}-airflow-worker-1"

    # Check if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping $PROJECT_NAME..."

        # Try docker compose down (suppress warnings, check if container still exists after)
        docker compose --project-name $PROJECT_NAME \
            --file docker-compose_worker.yaml \
            --file docker-compose_worker.dev.yaml \
            down 2>/dev/null

        # If container still exists, try prod config
        if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            docker compose --project-name $PROJECT_NAME \
                --file docker-compose_worker.yaml \
                --file docker-compose_worker.prod.yaml \
                down 2>/dev/null
        fi

        # If container still exists, force remove
        if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            docker stop $CONTAINER_NAME 2>/dev/null
            docker rm $CONTAINER_NAME 2>/dev/null
        fi
    fi
done

echo ""
echo "Checking remaining workers..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep worker || echo "All workers stopped."
