#!/bin/bash
# setup.sh - MLflow OIDC Docker Environment Setup Script
# Supports MLflow v3.8.1 with OIDC v5.7.0 (build 20251227)
# Using GitHub Container Registry (ghcr.io) with Keycloak for enterprise

set -e

echo "=========================================="
echo "MLflow OIDC Docker Environment Setup"
echo "Version: MLflow v3.8.1 + OIDC v5.7.0"
echo "=========================================="

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "docker-compose is not installed."
    exit 1
fi

# Use docker compose (v2) or docker-compose (v1)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Parse command line arguments
DEPLOY_MODE="oidc"
if [ "$1" == "--basic-auth" ]; then
    DEPLOY_MODE="basic"
    print_status "Deploying with Basic Auth only (no Keycloak)"
fi

# Build and start containers
print_status "Building and starting containers..."

if [ "$DEPLOY_MODE" == "basic" ]; then
    $COMPOSE_CMD --profile basic-auth up -d --build postgres mlflow
else
    $COMPOSE_CMD up -d --build postgres keycloak mlflow-oidc
fi

# Wait for services to be ready
print_status "Waiting for services to start..."

# Wait for PostgreSQL
print_status "Waiting for PostgreSQL..."
timeout=60
while [ $timeout -gt 0 ]; do
    if docker exec mlflow-postgres pg_isready -U mlflow -d mlflow_db > /dev/null 2>&1; then
        print_status "PostgreSQL is ready!"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    print_error "PostgreSQL did not start in time"
    exit 1
fi

# Wait for Keycloak (only in OIDC mode)
if [ "$DEPLOY_MODE" == "oidc" ]; then
    print_status "Waiting for Keycloak (this may take 2-3 minutes)..."
    timeout=180
    while [ $timeout -gt 0 ]; do
        if curl -sf http://localhost:8080/health/ready > /dev/null 2>&1; then
            print_status "Keycloak is ready!"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done

    if [ $timeout -le 0 ]; then
        print_warning "Keycloak health check timed out, but may still be starting..."
        print_warning "Check status with: docker compose logs keycloak"
    fi
fi

# Wait for MLflow
print_status "Waiting for MLflow OIDC Server..."
MLFLOW_CONTAINER="mlflow-oidc-server"
MLFLOW_PORT="5000"
if [ "$DEPLOY_MODE" == "basic" ]; then
    MLFLOW_CONTAINER="mlflow-server"
    MLFLOW_PORT="5001"
fi

timeout=90
while [ $timeout -gt 0 ]; do
    if curl -sf http://localhost:$MLFLOW_PORT/health > /dev/null 2>&1 || curl -sf http://localhost:$MLFLOW_PORT/ > /dev/null 2>&1; then
        print_status "MLflow is ready!"
        break
    fi
    sleep 3
    timeout=$((timeout - 3))
done

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Deployment Mode: $DEPLOY_MODE"
echo ""
echo "Services:"
echo "  - MLflow UI:     http://localhost:$MLFLOW_PORT"
if [ "$DEPLOY_MODE" == "oidc" ]; then
echo "  - Keycloak:      http://localhost:8080"
fi
echo "  - PostgreSQL:    localhost:5432"
echo ""
echo "Default Credentials:"
echo "  MLflow Admin:    admin / admin_password"
if [ "$DEPLOY_MODE" == "oidc" ]; then
echo "  Keycloak Admin:  admin / admin"
echo "  Test User:       mlflow-user / mlflow-password"
fi
echo ""
echo "GitHub Container: ghcr.io/mlflow/mlflow:v2.10.0"
echo ""
echo "Commands:"
echo "  View logs:       $COMPOSE_CMD logs -f"
echo "  Stop:            $COMPOSE_CMD down"
echo "  Stop + cleanup:  $COMPOSE_CMD down -v"
echo ""
