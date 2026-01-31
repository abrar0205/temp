# MLflow Docker Setup - Complete Verification Guide

This document provides step-by-step commands to verify the MLflow OIDC setup meets all DoD requirements.

## DoD Requirements

1. ✅ MLflow OIDC container running locally with PostgreSQL running in docker
2. ✅ Users can create access tokens via UI/CLI
3. ✅ Users can access experiment results using tokens *(primary goal)*

## Quick Start Commands

### Step 1: Setup Environment
```bash
cd Mlflow_docker
cp .env.example .env

# Generate and set secret key
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
sed -i "s/REPLACE_WITH_SECURE_RANDOM_KEY/$SECRET_KEY/" .env
```

### Step 2: Start All Services
```bash
docker compose up -d --build
```

### Step 3: Wait for Services (about 2 minutes)
```bash
# Check all containers are healthy
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected output:
```
NAMES             STATUS
mlflow-server     Up X minutes (healthy)
mlflow-keycloak   Up X minutes (healthy)
mlflow-postgres   Up X minutes (healthy)
mlflow-redis      Up X minutes (healthy)
mlflow-minio      Up X minutes (healthy)
```

### Step 4: Verify MLflow Health
```bash
curl http://localhost:5000/health
```
Expected: `OK`

### Step 5: Test Authentication (DoD #2)
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{"max_results": 100}'
```

### Step 6: Create Experiment
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{"name": "my-test-experiment"}'
```

### Step 7: Create a Run
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/create" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": "1"}'
```
Save the `run_id` from the response.

### Step 8: Log Metrics and Parameters
```bash
RUN_ID="<your-run-id-from-step-7>"

# Log parameter
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-parameter" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"key\": \"learning_rate\", \"value\": \"0.01\"}"

# Log metric
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"key\": \"accuracy\", \"value\": 0.95, \"timestamp\": $(date +%s)000}"
```

### Step 9: Finish the Run
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/update" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"status\": \"FINISHED\"}"
```

### Step 10: Retrieve Results (DoD #3 - PRIMARY GOAL)
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["1"], "max_results": 10}' | python3 -m json.tool
```

### Step 11: Verify PostgreSQL Storage (DoD #1)
```bash
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT * FROM experiments;"
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT run_uuid, status FROM runs;"
```

### Step 12: Run Automated Test Script
```bash
pip install requests
python3 test_mlflow_token_access.py
```

## Clean Up
```bash
docker compose down -v
```

## Services & Ports

| Service | Port | URL |
|---------|------|-----|
| MLflow | 5000 | http://localhost:5000 |
| Keycloak | 8080 | http://localhost:8080 |
| MinIO Console | 9001 | http://localhost:9001 |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |

## Default Credentials

- **MLflow**: admin / password
- **Keycloak Admin**: admin / admin123
- **MinIO**: minioadmin / minioadmin123
- **PostgreSQL**: mlflow / mlflow123
