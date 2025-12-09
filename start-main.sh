#!/bin/bash
# Start Airflow Main Server
# Usage: ./start-main.sh [dev|prod] [profile]
# Example: ./start-main.sh dev flower

set -e

ENV=${1:-dev}
PROFILE=${2:-}

echo "============================================"
echo "Starting Airflow Main Server"
echo "Environment: $ENV"
echo "============================================"

# Validate environment
if [[ "$ENV" != "dev" && "$ENV" != "prod" ]]; then
    echo "Error: Invalid environment. Use 'dev' or 'prod'"
    exit 1
fi

# Set env files
BASE_ENV="env/base.env"
ENV_FILE="env/${ENV}/main.env"

# Check if env files exist
if [[ ! -f "$BASE_ENV" ]]; then
    echo "Error: $BASE_ENV not found"
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found"
    exit 1
fi

# Build command
CMD="docker compose --file docker-compose_main.yaml"
CMD="$CMD --env-file $BASE_ENV"
CMD="$CMD --env-file $ENV_FILE"

# Add profile if specified
if [[ -n "$PROFILE" ]]; then
    CMD="$CMD --profile $PROFILE"
    echo "Profile: $PROFILE"
fi

CMD="$CMD up -d"

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================"
echo "Main server started successfully!"
echo "============================================"
echo ""
echo "Useful commands:"
echo "  View logs:    docker compose --file docker-compose_main.yaml logs -f"
echo "  Stop:         docker compose --file docker-compose_main.yaml down"
echo "  Web UI:       http://localhost:8080"
if [[ "$PROFILE" == "flower" ]]; then
    echo "  Flower:       http://localhost:5555"
fi
