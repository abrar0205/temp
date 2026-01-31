# MLflow OIDC Local Deployment - Complete Verification Guide

This guide provides **step-by-step commands** to verify the MLflow OIDC local deployment with PostgreSQL backend and token-based access.

---

## Quick Start (Copy-Paste Ready)

```bash
# Navigate to the Mlflow_docker directory
cd Mlflow_docker

# Step 1: Create .env file
cp .env.example .env

# Step 2: Start all services
docker compose up -d --build

# Step 3: Wait for services (about 60 seconds)
sleep 60

# Step 4: Verify everything is running
docker ps --format "table {{.Names}}\t{{.Status}}"

# Step 5: Test token authentication
curl -u admin:password http://localhost:5000/health
```

---

## Detailed Verification Steps

### Step 1: Prerequisites Check

```bash
# Verify Docker is installed
docker --version

# Verify Docker Compose is installed
docker compose version

# Verify you're in the right directory
ls -la
# Should see: docker-compose.yml, Dockerfile, .env.example, etc.
```

### Step 2: Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# Verify .env was created
cat .env
```

**Expected output:**
```
SECRET_KEY=REPLACE_ME_WITH_SECURE_SECRET_KEY_32_CHARS_MIN
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow123
...
```

### Step 3: Start Docker Services

```bash
# Build and start all containers
docker compose up -d --build

# This will start:
# - mlflow-postgres (PostgreSQL database)
# - mlflow-redis (Session storage)
# - mlflow-minio (S3-compatible artifact storage)
# - mlflow-keycloak (OIDC provider)
# - mlflow-server (MLflow tracking server)
```

**Wait 60-90 seconds** for all services to initialize.

### Step 4: Verify All Containers Are Running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Expected output:**
```
NAMES             STATUS                    PORTS
mlflow-server     Up X minutes (healthy)    0.0.0.0:5000->5000/tcp
mlflow-keycloak   Up X minutes (healthy)    0.0.0.0:8080->8080/tcp
mlflow-postgres   Up X minutes (healthy)    0.0.0.0:5432->5432/tcp
mlflow-redis      Up X minutes (healthy)    0.0.0.0:6379->6379/tcp
mlflow-minio      Up X minutes (healthy)    0.0.0.0:9000-9001->9000-9001/tcp
```

### Step 5: Verify MLflow Health

```bash
curl http://localhost:5000/health
```

**Expected output:**
```
OK
```

### Step 6: Test Token Authentication

```bash
# Access MLflow API with basic auth token
curl -u admin:password http://localhost:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"max_results": 10}'
```

**Expected output:**
```json
{
  "experiments": [
    {
      "experiment_id": "0",
      "name": "Default",
      ...
    }
  ]
}
```

---

## DoD Verification: Access Experiment Results Using Tokens

### DoD 1: Create an Experiment Using Token

```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{"name": "my-test-experiment"}'
```

**Expected output:**
```json
{"experiment_id": "1"}
```

### DoD 2: Create a Run and Log Metrics Using Token

```bash
# Create a run
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/create" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": "1"}'
```

**Expected output:**
```json
{
  "run": {
    "info": {
      "run_id": "abc123...",
      "experiment_id": "1",
      "status": "RUNNING"
    }
  }
}
```

Copy the `run_id` from the output and use it in the next commands:

```bash
# Log a metric (replace RUN_ID with actual run_id)
RUN_ID="your-run-id-here"

curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"key\": \"accuracy\", \"value\": 0.95, \"timestamp\": $(date +%s)000}"

# Log a parameter
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-parameter" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"key\": \"learning_rate\", \"value\": \"0.001\"}"

# Finish the run
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/update" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"status\": \"FINISHED\"}"
```

### DoD 3: Access Experiment Results Using Token (PRIMARY GOAL)

```bash
# List all experiments
curl -u admin:password "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Content-Type: application/json" \
  -d '{"max_results": 100}'

# Get runs with metrics and parameters
curl -u admin:password "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["1"], "max_results": 100}'
```

**Expected output shows metrics and parameters:**
```json
{
  "runs": [
    {
      "info": {
        "run_id": "...",
        "status": "FINISHED"
      },
      "data": {
        "metrics": [
          {"key": "accuracy", "value": 0.95}
        ],
        "params": [
          {"key": "learning_rate", "value": "0.001"}
        ]
      }
    }
  ]
}
```

---

## Verify PostgreSQL Backend Storage

```bash
# Connect to PostgreSQL and check experiments table
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT * FROM experiments;"

# Check runs table
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT run_uuid, experiment_id, status FROM runs;"

# Check metrics table
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT * FROM metrics;"

# Check params table
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT * FROM params;"
```

---

## Access UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| **MLflow UI** | http://localhost:5000 | admin / password |
| **Keycloak Admin** | http://localhost:8080 | admin / admin123 |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |

---

## Run Automated Test Script

```bash
# Run the Python test script
python3 test_mlflow_token_access.py
```

**Expected output:**
```
Testing MLflow Token-Based Access
==================================
✓ MLflow server is healthy
✓ Authentication successful
✓ Created experiment with ID: X
✓ Created run: abc123...
✓ Logged metrics and parameters
✓ Retrieved experiment results
✓ All tests passed!
```

---

## Cleanup

```bash
# Stop all containers
docker compose down

# Stop and remove all data (volumes)
docker compose down -v

# Remove images too
docker compose down -v --rmi local
```

---

## Troubleshooting

### Container not starting?
```bash
# Check logs
docker logs mlflow-server
docker logs mlflow-postgres
```

### Authentication failing?
```bash
# Default credentials are:
# Username: admin
# Password: password

# Test with curl
curl -v -u admin:password http://localhost:5000/health
```

### PostgreSQL connection issues?
```bash
# Check if PostgreSQL is healthy
docker exec mlflow-postgres pg_isready -U mlflow

# Check PostgreSQL logs
docker logs mlflow-postgres
```

---

## Summary: DoD Checklist

| Requirement | Command to Verify | Expected Result |
|-------------|-------------------|-----------------|
| MLflow container running | `docker ps \| grep mlflow-server` | Status: Up (healthy) |
| PostgreSQL running | `docker ps \| grep mlflow-postgres` | Status: Up (healthy) |
| Token auth works | `curl -u admin:password http://localhost:5000/health` | `OK` |
| Create experiment | `curl -u admin:password -X POST .../experiments/create` | `{"experiment_id": "X"}` |
| Access results | `curl -u admin:password .../runs/search` | JSON with metrics/params |

**All DoD requirements are verified when:**
1. ✅ All 5 containers show "healthy" status
2. ✅ `curl -u admin:password http://localhost:5000/health` returns `OK`
3. ✅ API calls with `-u admin:password` return experiment data
