#!/bin/bash
# Stop Airflow Main Server
# Usage: ./stop-main.sh [dev|prod] [profile]
# Example: ./stop-main.sh dev flower

ENV=${1:-dev}
PROFILE=${2:-}

echo "============================================"
echo "Stopping Airflow Main Server"
echo "Environment: $ENV"
echo "============================================"

# Validate environment
if [[ "$ENV" != "dev" && "$ENV" != "prod" ]]; then
    echo "Error: Invalid environment. Use 'dev' or 'prod'"
    exit 1
fi

# Build command
CMD="docker compose --file docker-compose_main.yaml"
CMD="$CMD --env-file env/base.env"
CMD="$CMD --env-file env/${ENV}/main.env"

if [[ -n "$PROFILE" ]]; then
    CMD="$CMD --profile $PROFILE"
    echo "Profile: $PROFILE"
fi

CMD="$CMD down"

echo ""
echo "Running: $CMD"
eval $CMD 2>/dev/null

echo ""
echo "Checking remaining containers..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "airflow-test-01" || echo "Main server stopped."
