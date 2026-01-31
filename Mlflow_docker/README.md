# MLflow OIDC Local Deployment with PostgreSQL

Local Docker deployment of MLflow with OIDC authentication, PostgreSQL backend, and token-based access.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MLflow UI     │────▶│  MLflow Server  │────▶│   PostgreSQL    │
│  localhost:5000 │     │   (with auth)   │     │  (tracking DB)  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Keycloak     │     │      MinIO      │     │      Redis      │
│  localhost:8080 │     │  (S3 artifacts) │     │   (sessions)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Quick Start

```bash
# 1. Create environment file
cp .env.example .env

# 2. Start all services
docker compose up -d --build

# 3. Wait for services (60 seconds)
sleep 60

# 4. Verify
curl -u admin:password http://localhost:5000/health
# Expected: OK
```

## Services

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | admin / password |
| Keycloak | http://localhost:8080 | admin / admin123 |
| MinIO | http://localhost:9001 | minioadmin / minioadmin123 |
| PostgreSQL | localhost:5432 | mlflow / mlflow123 |
| Redis | localhost:6379 | - |

## Verify Token-Based Access

### 1. Check Services Running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected: All 5 containers show "healthy"

### 2. Test Token Authentication

```bash
curl -u admin:password http://localhost:5000/health
```

Expected: `OK`

### 3. Create Experiment with Token

```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{"name": "my-experiment"}'
```

Expected: `{"experiment_id": "1"}`

### 4. Access Results with Token

```bash
curl -u admin:password \
  "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["1"]}'
```

Expected: JSON with runs, metrics, and parameters

### 5. Run Automated Test

```bash
python3 test_mlflow_token_access.py
```

## DoD Verification

| Requirement | Command | Expected |
|-------------|---------|----------|
| MLflow running | `docker ps \| grep mlflow-server` | Up (healthy) |
| PostgreSQL running | `docker ps \| grep mlflow-postgres` | Up (healthy) |
| Token auth works | `curl -u admin:password .../health` | OK |
| Create tokens | Use `admin:password` credentials | ✅ Works |
| Access results | API returns experiments/runs | ✅ Works |

## Cleanup

```bash
# Stop services (keep data)
docker compose stop

# Stop and remove (keep data)
docker compose down

# Remove everything including data
docker compose down -v
```

## Files

- `docker-compose.yml` - Service definitions
- `Dockerfile` - MLflow server image
- `.env.example` - Environment template
- `init-db.sh` - PostgreSQL initialization
- `configure-keycloak.sh` - Keycloak setup
- `test_mlflow_token_access.py` - Verification script
- `VERIFICATION_GUIDE.md` - Detailed step-by-step guide
