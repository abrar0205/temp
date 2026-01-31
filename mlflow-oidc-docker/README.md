# MLflow OIDC Docker Environment

This directory contains a complete Docker-based setup for MLflow with authentication using PostgreSQL as the tracking backend. The setup supports token-based access for programmatic interaction with MLflow.

## Quick Start

```bash
# Make setup script executable
chmod +x setup.sh

# Start the environment
./setup.sh

# Or manually with docker-compose
docker compose up -d --build
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Docker Network                             │
│                                                               │
│  ┌─────────────┐                      ┌─────────────┐        │
│  │   MLflow    │ PostgreSQL Backend   │ PostgreSQL  │        │
│  │   Server    │─────────────────────►│  Database   │        │
│  │  Port 5000  │                      │  Port 5432  │        │
│  └─────────────┘                      └─────────────┘        │
│        │                                                      │
│        │ Basic Auth (Token-based access)                     │
│        ▼                                                      │
│   REST API / UI                                               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| MLflow | 5000 | Tracking server with basic auth |
| PostgreSQL | 5432 | Tracking database backend |

## Default Credentials

### MLflow Server
- **Admin Username:** `admin`
- **Admin Password:** `admin_password`

## Verified Functionality

The setup has been verified to support:

✓ **Token-based authentication** - Basic auth tokens work for API access
✓ **Experiment creation** - Create experiments via API/CLI
✓ **Run logging** - Log parameters and metrics
✓ **Results retrieval** - Fetch experiment results using tokens
✓ **PostgreSQL backend** - Persistent storage with PostgreSQL

## Usage

### Access MLflow UI

1. Open http://localhost:5000 in your browser
2. Login with admin credentials: `admin` / `admin_password`

### Create and Use Access Tokens

#### Via CLI with Environment Variables

```bash
# Set environment variables for authentication
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=admin_password

# Test connection
mlflow experiments search
```

#### Via Python SDK

```python
import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set credentials via environment
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "admin_password"

# Create an experiment
mlflow.create_experiment("my-experiment")

# Log a run
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("accuracy", 0.95)
```

#### Via REST API with Basic Auth (Token-like access)

```bash
# Search experiments with basic auth
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -u admin:admin_password \
  -H "Content-Type: application/json" \
  -d '{"max_results": 100}'

# Create an experiment
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -u admin:admin_password \
  -H "Content-Type: application/json" \
  -d '{"name": "my-api-experiment"}'

# Create a run
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/create" \
  -u admin:admin_password \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": "0"}'

# Log a metric
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" \
  -u admin:admin_password \
  -H "Content-Type: application/json" \
  -d '{"run_id": "<your-run-id>", "key": "accuracy", "value": 0.95, "timestamp": 1234567890}'
```

### Test Script

Run the test script to verify token-based access:

```bash
# Install dependencies
pip install requests

# Run tests
python test_mlflow_access.py
```

Expected output:
```
============================================================
MLflow Access Token Test Script
============================================================
✓ MLflow server is healthy
✓ Authentication successful  
✓ Created experiment with ID: X
✓ Created run with ID: ...
✓ Logged params and metrics
✓ Run finished successfully
✓ Found X run(s) with metrics and parameters
✓ Found X experiment(s)
============================================================
✓ All tests passed! Token-based access is working correctly.
============================================================
```

## Managing Users

### Create New User via MLflow CLI

```bash
# Connect to MLflow container
docker exec -it mlflow-server bash

# Create a new user
mlflow users create -u newuser -p newpassword
```

## Container Management

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f mlflow
docker compose logs -f postgres
```

### Stop Services

```bash
# Stop (preserves data)
docker compose stop

# Stop and remove containers (preserves volumes)
docker compose down

# Full cleanup (removes everything including data)
docker compose down -v
```

### Restart Services

```bash
docker compose restart
docker compose restart mlflow
```

## Troubleshooting

### MLflow not starting

1. Check if PostgreSQL is ready:
   ```bash
   docker exec mlflow-postgres pg_isready -U mlflow
   ```

2. Check MLflow logs:
   ```bash
   docker compose logs mlflow
   ```

### Cannot authenticate

1. Verify credentials are correct
2. Check if MLflow is using basic-auth mode:
   ```bash
   docker exec mlflow-server cat /proc/1/cmdline | tr '\0' ' '
   ```

### Database connection issues

1. Check PostgreSQL is running:
   ```bash
   docker compose ps postgres
   ```

2. Test connection:
   ```bash
   docker exec -it mlflow-postgres psql -U mlflow -d mlflow_db -c "SELECT 1"
   ```

## File Structure

```
mlflow-oidc-docker/
├── docker-compose.yml        # Main compose configuration
├── Dockerfile.mlflow         # MLflow server image
├── setup.sh                  # Automated setup script
├── test_mlflow_access.py     # Token access test script
├── README.md                 # This file
└── mlflow_config/
    └── auth_config.ini       # MLflow auth configuration
```

## Security Notes

⚠️ **For Production Use:**

1. Change all default passwords
2. Enable HTTPS/TLS
3. Use proper secret management
4. Configure network policies
5. Enable audit logging
6. Set up proper backup procedures

## Extending with OIDC

For full OIDC authentication with providers like Keycloak, Azure AD, or Okta:

1. Add a Keycloak service to docker-compose.yml
2. Configure MLflow with OIDC environment variables
3. Set up realm and client in Keycloak
4. Update redirect URIs

See the [MLflow Authentication documentation](https://mlflow.org/docs/latest/auth/index.html) for more details.
