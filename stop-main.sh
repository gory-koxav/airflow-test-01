#!/bin/bash
# Stop Airflow Main Server
# Usage: ./stop-main.sh

echo "Stopping Airflow Main Server..."

docker compose --file docker-compose_main.yaml down

echo "Main server stopped."
