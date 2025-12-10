#!/bin/bash
# Stop all Airflow Workers
# Usage: ./stop-workers.sh [dev|prod] [worker_count]
# Example: ./stop-workers.sh dev 4

ENV=${1:-dev}
WORKER_COUNT=${2:-10}

echo "============================================"
echo "Stopping Airflow Workers"
echo "Environment: $ENV"
echo "============================================"

# Validate environment
if [[ "$ENV" != "dev" && "$ENV" != "prod" ]]; then
    echo "Error: Invalid environment. Use 'dev' or 'prod'"
    exit 1
fi

OVERRIDE_FILE="docker-compose_worker.${ENV}.yaml"

for i in $(seq 1 $WORKER_COUNT); do
    PROJECT_NAME="airflow-worker-${i}"
    CONTAINER_NAME="${PROJECT_NAME}-airflow-worker-1"

    # Check if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping $PROJECT_NAME..."

        # Try docker compose down with specified environment
        docker compose --project-name $PROJECT_NAME \
            --file docker-compose_worker.yaml \
            --file $OVERRIDE_FILE \
            down 2>/dev/null

        # If container still exists, try the other environment config
        if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            OTHER_ENV=$([[ "$ENV" == "dev" ]] && echo "prod" || echo "dev")
            docker compose --project-name $PROJECT_NAME \
                --file docker-compose_worker.yaml \
                --file docker-compose_worker.${OTHER_ENV}.yaml \
                down 2>/dev/null
        fi

        # If container still exists, force remove
        if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo "  Force removing $CONTAINER_NAME..."
            docker stop $CONTAINER_NAME 2>/dev/null
            docker rm $CONTAINER_NAME 2>/dev/null
        fi
    fi
done

echo ""
echo "Checking remaining workers..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep worker || echo "All workers stopped."
