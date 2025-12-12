#!/bin/bash
# Stop Airflow Workers
# Usage: ./stop-workers.sh [dev|prod] [type] [bay_ids...]
#
# Examples:
#   ./stop-workers.sh dev                    # Stop all workers
#   ./stop-workers.sh dev obs                # Stop all observer workers
#   ./stop-workers.sh dev proc               # Stop all processor workers
#   ./stop-workers.sh dev obs 64bay          # Stop observer for 64bay only
#   ./stop-workers.sh dev all 64bay 12bay    # Stop all workers for 64bay and 12bay

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
echo -e "${CYAN}Stopping Airflow Workers${NC}"
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

OVERRIDE_FILE="docker-compose_worker.${ENV}.yaml"
WORKER_ENV_DIR="env/${ENV}"
STOPPED_WORKERS=()

stop_worker() {
    local WORKER_ENV=$1
    local WORKER_NAME=$(grep "^WORKER_NAME=" "$WORKER_ENV" | cut -d'=' -f2)
    local PROJECT_NAME="airflow-${WORKER_NAME}"
    local CONTAINER_NAME="${PROJECT_NAME}-airflow-worker-1"

    # Check if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Stopping: ${WORKER_NAME}${NC}"

        # Try docker compose down
        docker compose --project-name $PROJECT_NAME \
            --file docker-compose_worker.yaml \
            --file $OVERRIDE_FILE \
            down 2>/dev/null

        # If container still exists, force remove
        if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo "  Force removing $CONTAINER_NAME..."
            docker stop $CONTAINER_NAME 2>/dev/null || true
            docker rm $CONTAINER_NAME 2>/dev/null || true
        fi

        STOPPED_WORKERS+=("$WORKER_NAME")
        echo -e "  ${GREEN}✓${NC} Stopped"
    fi
}

# Find and stop workers based on env files
for WORKER_ENV in "$WORKER_ENV_DIR"/worker-*.env; do
    [[ -f "$WORKER_ENV" ]] || continue

    # Extract type and bay from filename: worker-{type}-{bay}.env
    FILENAME=$(basename "$WORKER_ENV")
    NAME_PART=${FILENAME#worker-}
    NAME_PART=${NAME_PART%.env}

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

    stop_worker "$WORKER_ENV"
done

# Also stop any legacy workers (airflow-worker-N pattern) if stopping all
if [[ "$TYPE" == "all" && ${#BAY_IDS[@]} -eq 0 ]]; then
    echo ""
    echo "Checking for legacy workers (airflow-worker-N pattern)..."
    for i in $(seq 1 10); do
        LEGACY_PROJECT="airflow-worker-${i}"
        LEGACY_CONTAINER="${LEGACY_PROJECT}-airflow-worker-1"

        if docker ps -a --format "{{.Names}}" | grep -q "^${LEGACY_CONTAINER}$"; then
            echo -e "${YELLOW}Stopping legacy worker: ${LEGACY_PROJECT}${NC}"
            docker compose --project-name $LEGACY_PROJECT \
                --file docker-compose_worker.yaml \
                --file $OVERRIDE_FILE \
                down 2>/dev/null || true

            if docker ps -a --format "{{.Names}}" | grep -q "^${LEGACY_CONTAINER}$"; then
                docker stop $LEGACY_CONTAINER 2>/dev/null || true
                docker rm $LEGACY_CONTAINER 2>/dev/null || true
            fi
            STOPPED_WORKERS+=("$LEGACY_PROJECT")
        fi
    done
fi

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}Stop Summary${NC}"
echo -e "${CYAN}============================================${NC}"

if [[ ${#STOPPED_WORKERS[@]} -eq 0 ]]; then
    echo "No workers were running."
else
    echo "Stopped workers:"
    for WORKER in "${STOPPED_WORKERS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $WORKER"
    done
fi

echo ""
echo "Remaining worker containers:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "airflow-.*worker" || echo "  (none)"
