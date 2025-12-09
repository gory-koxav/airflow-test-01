#!/bin/bash
# Stop Airflow Main Server
# Usage: ./stop-main.sh [profile]
# Example: ./stop-main.sh flower

PROFILE=${1:-}

echo "Stopping Airflow Main Server..."

# Build command
CMD="docker compose --file docker-compose_main.yaml"
CMD="$CMD --env-file env/base.env"
CMD="$CMD --env-file env/dev/main.env"

if [[ -n "$PROFILE" ]]; then
    CMD="$CMD --profile $PROFILE"
    echo "Profile: $PROFILE"
fi

CMD="$CMD down"

eval $CMD 2>/dev/null

echo ""
echo "Checking remaining containers..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "airflow-test-01" || echo "Main server stopped."
