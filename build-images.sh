#!/bin/bash
# Build custom Airflow worker images
# Usage: ./build-images.sh [observer|processor|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

build_observer() {
    echo -e "${YELLOW}Building Observer image (GPU + ML packages)...${NC}"
    echo "This may take a while due to large ML packages (PyTorch, CUDA, etc.)"
    docker build \
        -f Dockerfile.observer \
        -t airflow-observer:latest \
        .
    echo -e "${GREEN}✓ Observer image built successfully${NC}"
}

build_processor() {
    echo -e "${YELLOW}Building Processor image (lightweight)...${NC}"
    docker build \
        -f Dockerfile.processor \
        -t airflow-processor:latest \
        .
    echo -e "${GREEN}✓ Processor image built successfully${NC}"
}

case "${1:-all}" in
    observer)
        build_observer
        ;;
    processor)
        build_processor
        ;;
    all)
        build_observer
        build_processor
        echo ""
        echo -e "${GREEN}All images built successfully!${NC}"
        echo ""
        echo "Available images:"
        docker images | grep -E "airflow-(observer|processor)"
        ;;
    *)
        echo "Usage: $0 [observer|processor|all]"
        echo ""
        echo "  observer  - Build Observer image (GPU + ML packages)"
        echo "  processor - Build Processor image (lightweight)"
        echo "  all       - Build both images (default)"
        exit 1
        ;;
esac
