# MLflow OIDC Docker Environment

This directory contains a complete Docker-based setup for MLflow with OIDC authentication using PostgreSQL as the tracking backend. The setup uses the official MLflow GitHub container with Keycloak for enterprise OIDC authentication.

## Version Information

- **MLflow:** v3.8.1 (via ghcr.io/mlflow/mlflow)
- **OIDC:** v5.7.0 (build 20251227)
- **Keycloak:** 23.0 (OIDC Provider)
- **PostgreSQL:** 15-alpine

## Quick Start

```bash
# Make setup script executable
chmod +x setup.sh

# Start the environment with OIDC (Keycloak)
./setup.sh

# Or manually with docker compose
docker compose up -d --build
```

### Alternative: Basic Auth Only (without Keycloak)

```bash
# Start with basic auth profile (simpler setup)
docker compose --profile basic-auth up -d postgres mlflow
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Docker Network                                 │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Keycloak   │    │   MLflow    │    │ PostgreSQL  │          │
│  │   (OIDC)    │◄───│   OIDC      │───►│  Database   │          │
│  │  Port 8080  │    │  Port 5000  │    │  Port 5432  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        │                   │                                      │
│        │ OIDC Auth         │ Token/Basic Auth                    │
│        ▼                   ▼                                      │
│   Admin Console       REST API / UI                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| MLflow OIDC | 5000 | Tracking server with OIDC/Basic auth |
| Keycloak | 8080 | Enterprise OIDC identity provider |
| PostgreSQL | 5432 | Tracking database backend |

## Default Credentials

### MLflow Server (Basic Auth)
- **Admin Username:** `admin`
- **Admin Password:** `admin_password`

### Keycloak Admin Console
- **URL:** http://localhost:8080
- **Username:** `admin`
- **Password:** `admin`

### Keycloak Test User
- **Username:** `mlflow-user`
- **Password:** `mlflow-password`

## Verified Functionality

The setup has been verified to support:

✓ **OIDC Authentication** - Enterprise SSO via Keycloak
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
├── docker-compose.yml        # Main compose configuration (OIDC + Basic Auth)
├── Dockerfile.mlflow         # MLflow server image (basic auth fallback)
├── setup.sh                  # Automated setup script
├── test_mlflow_access.py     # Token access test script
├── README.md                 # This file
├── keycloak/
│   └── realm-export.json     # Keycloak realm with MLflow client
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
7. Configure Keycloak with proper realm settings
8. Use secure OIDC client secrets

## OIDC Configuration

### Keycloak Realm Settings

The default Keycloak realm (`mlflow`) is pre-configured with:
- Client ID: `mlflow-client`
- Client Secret: `mlflow-secret`
- Redirect URIs: `http://localhost:5000/*`
- Test user: `mlflow-user` / `mlflow-password`

### Customizing OIDC Settings

To use a different OIDC provider or customize settings:

1. Update environment variables in `docker-compose.yml`:
   ```yaml
   OIDC_PROVIDER_URL: https://your-oidc-provider.com/realms/your-realm
   OIDC_CLIENT_ID: your-client-id
   OIDC_CLIENT_SECRET: your-client-secret
   ```

2. Or modify the Keycloak realm export in `keycloak/realm-export.json`

## GitHub Container Registry

The MLflow OIDC image is pulled from GitHub Container Registry:

```bash
# Pull the latest MLflow image
docker pull ghcr.io/mlflow/mlflow:v2.10.0

# For enterprise OIDC builds (when available)
# docker pull ghcr.io/mlflow/mlflow-oidc:v5.7.0
```

## References

- [MLflow Authentication Documentation](https://mlflow.org/docs/latest/auth/index.html)
- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [MLflow GitHub Container Registry](https://github.com/mlflow/mlflow/pkgs/container/mlflow)
