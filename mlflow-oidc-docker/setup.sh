#!/bin/bash
# setup.sh - MLflow OIDC Docker Environment Setup Script

set -e

echo "=========================================="
echo "MLflow OIDC Docker Environment Setup"
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

# Build and start containers
print_status "Building and starting containers..."

# Use docker compose (v2) or docker-compose (v1)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

$COMPOSE_CMD up -d --build

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

# Wait for Keycloak
print_status "Waiting for Keycloak (this may take a minute)..."
timeout=120
while [ $timeout -gt 0 ]; do
    if curl -s http://localhost:8080/health/ready > /dev/null 2>&1; then
        print_status "Keycloak is ready!"
        break
    fi
    sleep 5
    timeout=$((timeout - 5))
done

if [ $timeout -le 0 ]; then
    print_warning "Keycloak health check timed out, but may still be starting..."
fi

# Wait for MLflow
print_status "Waiting for MLflow..."
timeout=60
while [ $timeout -gt 0 ]; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1 || curl -s http://localhost:5000/ > /dev/null 2>&1; then
        print_status "MLflow is ready!"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - MLflow UI:     http://localhost:5000"
echo "  - Keycloak:      http://localhost:8080"
echo "  - PostgreSQL:    localhost:5432"
echo ""
echo "Default Credentials:"
echo "  MLflow Admin:    admin / admin_password"
echo "  Keycloak Admin:  admin / admin"
echo "  Test User:       mlflow-user / mlflow-password"
echo ""
echo "To view logs:      $COMPOSE_CMD logs -f"
echo "To stop:           $COMPOSE_CMD down"
echo "To stop + cleanup: $COMPOSE_CMD down -v"
echo ""
