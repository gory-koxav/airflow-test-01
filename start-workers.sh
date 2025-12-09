#!/bin/bash
# Start Airflow Workers
# Usage: ./start-workers.sh [dev|prod] [worker_count]
# Example: ./start-workers.sh dev 2

set -e

ENV=${1:-dev}
WORKER_COUNT=${2:-2}

echo "============================================"
echo "Starting Airflow Workers"
echo "Environment: $ENV"
echo "Worker Count: $WORKER_COUNT"
echo "============================================"

# Validate environment
if [[ "$ENV" != "dev" && "$ENV" != "prod" ]]; then
    echo "Error: Invalid environment. Use 'dev' or 'prod'"
    exit 1
fi

# Set env files
BASE_ENV="env/base.env"
COMPOSE_FILE="docker-compose_worker.yaml"
OVERRIDE_FILE="docker-compose_worker.${ENV}.yaml"

# Check if env files exist
if [[ ! -f "$BASE_ENV" ]]; then
    echo "Error: $BASE_ENV not found"
    exit 1
fi

if [[ ! -f "$OVERRIDE_FILE" ]]; then
    echo "Error: $OVERRIDE_FILE not found"
    exit 1
fi

# Start workers
for i in $(seq 1 $WORKER_COUNT); do
    # Determine worker env file
    if [[ $i -eq 1 ]]; then
        WORKER_ENV="env/${ENV}/worker.env"
    else
        WORKER_ENV="env/${ENV}/worker${i}.env"
    fi

    # Check if worker env file exists
    if [[ ! -f "$WORKER_ENV" ]]; then
        echo "Warning: $WORKER_ENV not found, skipping worker $i"
        continue
    fi

    echo ""
    echo "Starting Worker $i..."
    echo "  Env file: $WORKER_ENV"

    CMD="docker compose --project-name airflow-worker-${i}"
    CMD="$CMD --file $COMPOSE_FILE"
    CMD="$CMD --file $OVERRIDE_FILE"
    CMD="$CMD --env-file $BASE_ENV"
    CMD="$CMD --env-file $WORKER_ENV"
    CMD="$CMD up -d"

    echo "  Running: $CMD"
    eval $CMD
done

echo ""
echo "Waiting for workers to start..."
sleep 5

echo ""
echo "============================================"
echo "Worker Status:"
echo "============================================"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "airflow-worker" || echo "No workers found"

echo ""
echo "============================================"
echo "Workers started successfully!"
echo "============================================"
echo ""
echo "Useful commands:"
echo "  View logs:    docker logs airflow-worker-1-airflow-worker-1"
echo "  Stop all:     ./stop-workers.sh"
