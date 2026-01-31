#!/bin/bash
# setup.sh - MLflow OIDC Docker Environment Setup Script
# Based on mlflow-tracking-server-docker architecture
#
# Architecture:
# - MLflow Server with OIDC authentication
# - Keycloak OpenID Connect provider
# - PostgreSQL backend storage
# - Redis session storage
# - MinIO S3-compatible artifact storage

set -e

echo "=========================================="
echo "MLflow OIDC Docker Environment Setup"
echo "=========================================="

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    print_error "docker compose is not available."
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning "Please edit .env file with secure passwords before continuing!"
        print_warning "Generate a secret key with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
        echo ""
        read -p "Press Enter to continue with default values (not recommended for production)..."
    else
        print_error ".env.example not found. Cannot create .env file."
        exit 1
    fi
fi

# Load environment variables
set -a
source .env
set +a

print_status "Building and starting containers..."

# Build and start all services
docker compose up -d --build

# Wait for services to be ready
print_status "Waiting for services to start..."

# Wait for PostgreSQL
print_status "Waiting for PostgreSQL..."
timeout=60
while [ $timeout -gt 0 ]; do
    if docker exec mlflow-postgres pg_isready -U "${POSTGRES_USER:-mlflow}" -d "${POSTGRES_DB:-mlflow}" > /dev/null 2>&1; then
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

# Wait for Redis
print_status "Waiting for Redis..."
timeout=30
while [ $timeout -gt 0 ]; do
    # Use --no-auth-warning to suppress password exposure warning
    if docker exec mlflow-redis redis-cli --no-auth-warning -a "${REDIS_PASSWORD:-redis_password}" ping > /dev/null 2>&1; then
        print_status "Redis is ready!"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    print_warning "Redis health check timed out"
fi

# Wait for MinIO
print_status "Waiting for MinIO..."
timeout=30
while [ $timeout -gt 0 ]; do
    if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        print_status "MinIO is ready!"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    print_warning "MinIO health check timed out"
fi

# Wait for Keycloak
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
fi

# Wait for MLflow
print_status "Waiting for MLflow..."
timeout=90
while [ $timeout -gt 0 ]; do
    if curl -sf http://localhost:5000/health > /dev/null 2>&1 || curl -sf http://localhost:5000/ > /dev/null 2>&1; then
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
echo "Services:"
echo "  - MLflow UI:      http://localhost:5000"
echo "  - Keycloak:       http://localhost:8080"
echo "  - MinIO Console:  http://localhost:9001"
echo "  - PostgreSQL:     localhost:5432"
echo "  - Redis:          localhost:6379"
echo ""
echo "Next Steps:"
echo "  1. Configure Keycloak by running:"
echo "     ./configure-keycloak.sh"
echo ""
echo "  2. Update .env with the OIDC_CLIENT_SECRET from the script output"
echo ""
echo "  3. Restart MLflow:"
echo "     docker compose restart mlflow"
echo ""
echo "Default Credentials (from .env):"
echo "  Keycloak Admin:  ${KEYCLOAK_ADMIN:-admin} / ${KEYCLOAK_ADMIN_PASSWORD:-admin}"
echo "  MinIO Console:   ${MINIO_ROOT_USER:-minioadmin} / ${MINIO_ROOT_PASSWORD:-minioadmin_password}"
echo ""
echo "Commands:"
echo "  View logs:       docker compose logs -f"
echo "  Stop:            docker compose down"
echo "  Stop + cleanup:  docker compose down -v"
echo ""
