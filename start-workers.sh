#!/bin/bash
# Start Airflow Workers
# Usage: ./start-workers.sh [dev|prod] [type] [bay_ids...]
#
# Examples:
#   ./start-workers.sh dev                    # Start all workers (obs + proc) for all bays
#   ./start-workers.sh dev obs                # Start all observer workers
#   ./start-workers.sh dev proc               # Start all processor workers
#   ./start-workers.sh dev obs 64bay          # Start observer for 64bay only
#   ./start-workers.sh dev all 64bay 12bay    # Start all workers for 64bay and 12bay

set -e

ENV=${1:-dev}
TYPE=${2:-all}  # all, obs, proc

# Shift arguments safely
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
BAY_IDS=("$@")  # Remaining arguments are bay IDs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}Starting Airflow Workers${NC}"
echo -e "${CYAN}============================================${NC}"
echo "Environment: $ENV"
echo "Type: $TYPE"
if [[ ${#BAY_IDS[@]} -gt 0 ]]; then
    echo "Bay IDs: ${BAY_IDS[*]}"
else
    echo "Bay IDs: all"
fi
echo ""

# Validate environment
if [[ "$ENV" != "dev" && "$ENV" != "prod" ]]; then
    echo -e "${RED}Error: Invalid environment. Use 'dev' or 'prod'${NC}"
    exit 1
fi

# Validate type
if [[ "$TYPE" != "all" && "$TYPE" != "obs" && "$TYPE" != "proc" ]]; then
    echo -e "${RED}Error: Invalid type. Use 'all', 'obs', or 'proc'${NC}"
    exit 1
fi

# Set env files
BASE_ENV="env/base.env"
COMPOSE_FILE="docker-compose_worker.yaml"
OVERRIDE_FILE="docker-compose_worker.${ENV}.yaml"

# Check if env files exist
if [[ ! -f "$BASE_ENV" ]]; then
    echo -e "${RED}Error: $BASE_ENV not found${NC}"
    exit 1
fi

if [[ ! -f "$OVERRIDE_FILE" ]]; then
    echo -e "${RED}Error: $OVERRIDE_FILE not found${NC}"
    exit 1
fi

# Find matching worker env files
# Naming convention: worker-{type}-{bay_id}.env (e.g., worker-obs-64bay.env, worker-proc-64bay.env)
WORKER_ENV_DIR="env/${ENV}"
STARTED_WORKERS=()

start_worker() {
    local WORKER_ENV=$1
    local WORKER_NAME=$(grep "^WORKER_NAME=" "$WORKER_ENV" | cut -d'=' -f2)
    local WORKER_QUEUES=$(grep "^WORKER_QUEUES=" "$WORKER_ENV" | cut -d'=' -f2)
    local WORKER_IMAGE=$(grep "^AIRFLOW_IMAGE_NAME=" "$WORKER_ENV" | cut -d'=' -f2)
    local WORKER_GPU=$(grep "^WORKER_GPU=" "$WORKER_ENV" | cut -d'=' -f2)

    # Use WORKER_NAME as project name for unique container naming
    local PROJECT_NAME="airflow-${WORKER_NAME}"

    echo ""
    echo -e "${YELLOW}Starting Worker: ${WORKER_NAME}${NC}"
    echo "  Env file: $WORKER_ENV"
    echo "  Image: $WORKER_IMAGE"
    echo "  Queues: $WORKER_QUEUES"
    echo "  GPU: ${WORKER_GPU:-false}"

    CMD="docker compose --project-name ${PROJECT_NAME}"
    CMD="$CMD --file $COMPOSE_FILE"
    CMD="$CMD --file $OVERRIDE_FILE"

    # Add GPU override if WORKER_GPU=true
    if [[ "$WORKER_GPU" == "true" ]]; then
        GPU_OVERRIDE_FILE="docker-compose_worker.gpu.yaml"
        if [[ -f "$GPU_OVERRIDE_FILE" ]]; then
            CMD="$CMD --file $GPU_OVERRIDE_FILE"
            echo -e "  ${GREEN}GPU enabled via $GPU_OVERRIDE_FILE${NC}"
        else
            echo -e "  ${RED}Warning: GPU requested but $GPU_OVERRIDE_FILE not found${NC}"
        fi
    fi

    CMD="$CMD --env-file $BASE_ENV"
    CMD="$CMD --env-file $WORKER_ENV"
    CMD="$CMD up -d"

    echo "  Running: $CMD"
    eval $CMD

    STARTED_WORKERS+=("$WORKER_NAME:$WORKER_QUEUES")
}

# Find and start workers
for WORKER_ENV in "$WORKER_ENV_DIR"/worker-*.env; do
    [[ -f "$WORKER_ENV" ]] || continue

    # Extract type and bay from filename: worker-{type}-{bay}.env
    FILENAME=$(basename "$WORKER_ENV")
    # Remove 'worker-' prefix and '.env' suffix
    NAME_PART=${FILENAME#worker-}
    NAME_PART=${NAME_PART%.env}

    # Split by first '-' to get type and bay
    FILE_TYPE=${NAME_PART%%-*}
    FILE_BAY=${NAME_PART#*-}

    # Filter by type
    if [[ "$TYPE" != "all" && "$FILE_TYPE" != "$TYPE" ]]; then
        continue
    fi

    # Filter by bay IDs if specified
    if [[ ${#BAY_IDS[@]} -gt 0 ]]; then
        MATCH=false
        for BAY in "${BAY_IDS[@]}"; do
            if [[ "$FILE_BAY" == "$BAY" ]]; then
                MATCH=true
                break
            fi
        done
        if [[ "$MATCH" == false ]]; then
            continue
        fi
    fi

    start_worker "$WORKER_ENV"
done

# Check if any workers were started
if [[ ${#STARTED_WORKERS[@]} -eq 0 ]]; then
    echo ""
    echo -e "${RED}No matching worker env files found!${NC}"
    echo ""
    echo "Expected file naming convention: worker-{type}-{bay}.env"
    echo "  Examples:"
    echo "    env/${ENV}/worker-obs-64bay.env   (Observer for 64bay)"
    echo "    env/${ENV}/worker-proc-64bay.env  (Processor for 64bay)"
    echo ""
    echo "Available worker env files:"
    ls -1 "$WORKER_ENV_DIR"/worker*.env 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo ""
echo "Waiting for workers to start..."
sleep 5

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}Worker Status:${NC}"
echo -e "${CYAN}============================================${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep -E "airflow-" | grep -v "postgres\|redis\|scheduler\|apiserver" || echo "No workers found"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Workers started successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Started Workers:"
for WORKER_INFO in "${STARTED_WORKERS[@]}"; do
    WORKER_NAME=${WORKER_INFO%%:*}
    WORKER_QUEUES=${WORKER_INFO#*:}
    echo -e "  ${GREEN}✓${NC} $WORKER_NAME -> $WORKER_QUEUES"
done

echo ""
echo "Useful commands:"
echo "  View logs:    docker logs airflow-{worker_name}-airflow-worker-1"
echo "  Stop all:     ./stop-workers.sh $ENV"
echo "  Stop type:    ./stop-workers.sh $ENV obs"
echo "  Stop specific: ./stop-workers.sh $ENV obs 64bay"
