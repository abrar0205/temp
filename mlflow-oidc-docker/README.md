# MLflow with Keycloak OIDC Authentication

MLflow deployment with Keycloak SSO authentication, based on [mlflow-tracking-server-docker](https://github.com/mlflow-oidc/mlflow-tracking-server-docker).

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Docker Network                                   │
│                                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Keycloak   │    │   MLflow    │    │ PostgreSQL  │    │    Redis    │   │
│  │   (OIDC)    │◄───│   Server    │───►│  Database   │    │  (Sessions) │   │
│  │  Port 8080  │    │  Port 5000  │    │  Port 5432  │    │  Port 6379  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        │                   │                                     ▲           │
│        │                   │                                     │           │
│        │ OIDC Auth         │ S3 API                             │           │
│        │                   ▼                                     │           │
│        │            ┌─────────────┐                              │           │
│        │            │    MinIO    │                              │           │
│        │            │ (Artifacts) │                              │           │
│        │            │ Port 9000   │                              │           │
│        │            │ Port 9001   │ (Console)                    │           │
│        │            └─────────────┘                              │           │
│        │                                                         │           │
│        └─────────────────────────────────────────────────────────┘           │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Components:**

- **MLflow Server**: Tracking server with OIDC authentication (mlflow-oidc-auth)
- **Keycloak**: OpenID Connect provider for SSO
- **PostgreSQL**: Backend storage for MLflow and Keycloak
- **Redis**: Session storage
- **MinIO**: S3-compatible artifact storage

## Building the Docker Image

The MLflow OIDC server image is built from the Dockerfile based on [mlflow-tracking-server-docker](https://github.com/mlflow-oidc/mlflow-tracking-server-docker).

### Build Command

```bash
docker build --build-arg UV_HTTP_TIMEOUT=300 -t mlflow-oidc-server:latest .
```

**Command Explanation:**

- `docker build` - Build a Docker image from a Dockerfile
- `--build-arg UV_HTTP_TIMEOUT=300` - Set timeout for package installation to 300 seconds
  - Prevents build failures when downloading large packages (mlflow, dependencies)
  - 300 seconds (5 minutes) allows sufficient time for slow network connections
- `-t mlflow-oidc-server:latest` - Tag the image
- `.` - Build context (current directory contains the Dockerfile)

### What Gets Installed

The Dockerfile installs:

- Python 3.12
- MLflow 2.10.0
- mlflow-oidc-auth 0.3.0 (OIDC authentication plugin)
- psycopg2 (PostgreSQL driver)
- boto3 (S3/MinIO client)
- redis (Redis client for sessions)

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Generate a secure secret key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Edit .env and update:
# - SECRET_KEY (use generated value above)
# - All passwords (POSTGRES_PASSWORD, KEYCLOAK_ADMIN_PASSWORD, etc.)
```

### 2. Start Services

```bash
docker compose up -d
```

### 3. Configure Keycloak

```bash
# Wait for all services to be healthy (takes ~60-120 seconds)
docker compose ps

# Run configuration script to set up realm, client, and test user
chmod +x configure-keycloak.sh
./configure-keycloak.sh
```

The script will output the `OIDC_CLIENT_SECRET`. Update your `.env` file with this value and restart MLflow:

```bash
# Update OIDC_CLIENT_SECRET in .env file
# Then restart MLflow
docker compose restart mlflow
```

### 4. Access MLflow

| Service | URL | Description |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | Tracking server |
| Keycloak Admin | http://localhost:8080 | OIDC provider admin |
| MinIO Console | http://localhost:9001 | Artifact storage admin |

**MLflow Default Admin** (for basic-auth):
- Username: `admin`
- Password: `password` (default, change for production)

**Default Keycloak test user** (created by `configure-keycloak.sh`):
- Username: `mlflow_user`
- Password: `password123`

## Configuration

### Environment Variables

See `.env.example` for all options. Required variables:

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | Session encryption key (32+ random characters) |
| `OIDC_CLIENT_SECRET` | From Keycloak after running `configure-keycloak.sh` |
| `POSTGRES_PASSWORD` | PostgreSQL password for MLflow |
| `KEYCLOAK_ADMIN_PASSWORD` | Keycloak admin password |
| `REDIS_PASSWORD` | Redis password for sessions |
| `MINIO_ROOT_PASSWORD` | MinIO admin password |

### User Groups

Users must belong to one of these groups (configured in Keycloak):

- `mlflow_admins`: Full administrative access
- `mlflow_users`: Regular user access

Update group names via `OIDC_ADMIN_GROUP_NAME` and `OIDC_GROUP_NAME` in `.env`.

## Management

### View Logs

```bash
docker compose logs -f mlflow
docker compose logs -f keycloak
docker compose logs -f postgres
docker compose logs -f redis
docker compose logs -f minio
```

### Stop Services

```bash
docker compose down
```

### Reset All Data

```bash
docker compose down -v  # WARNING: Deletes all data
```

### Add New Users

1. Access Keycloak admin console: http://localhost:8080
2. Login with admin credentials from `.env`
3. Navigate to MLflow realm → Users → Add User
4. Set password in Credentials tab
5. Add user to `mlflow_admins` or `mlflow_users` group in Groups tab

## Troubleshooting

### "Invalid scopes" error

- Ensure `.env` has `OIDC_SCOPE` (singular, not `SCOPES`)
- Value should be: `"openid profile email"`

### "Session error: Missing OAuth state"

- Check Redis is running: `docker compose ps redis`
- Verify Redis connection: `docker exec mlflow-redis redis-cli -a <password> ping`

### "Authorization error: User is not allowed to login"

- User must be in `mlflow_admins` or `mlflow_users` group in Keycloak
- Check group membership in Keycloak admin console
- Verify `OIDC_ADMIN_GROUP_NAME` and `OIDC_GROUP_NAME` in `.env` match Keycloak groups

### "Internal Server Error"

- Check MLflow logs: `docker compose logs mlflow`
- Verify all environment variables are set correctly in `.env`

### MinIO connection issues

- Check MinIO is running: `docker compose ps minio`
- Verify credentials match in `.env` (`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`)
- Test MinIO API: `curl http://localhost:9000/minio/health/live`

## Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Service definitions |
| `.env` | Environment configuration (DO NOT COMMIT) |
| `.env.example` | Template for `.env` file |
| `Dockerfile.mlflow` | MLflow OIDC server image |
| `init-db.sh` | PostgreSQL initialization script |
| `configure-keycloak.sh` | Keycloak setup automation script |
| `test_mlflow_access.py` | Token access test script |

## Using with Python SDK

```python
import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# For basic auth (after OIDC login, use generated token)
os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_password"

# Create an experiment
mlflow.create_experiment("my-experiment")

# Log a run
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")  # Stored in MinIO
```

## Using with CLI

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_password

# List experiments
mlflow experiments search

# Create experiment
mlflow experiments create -n "my-experiment"
```

## Support

For issues with:

- **MLflow OIDC**: https://github.com/mlflow-oidc/mlflow-oidc-auth
- **MLflow**: https://github.com/mlflow/mlflow
- **Keycloak**: https://www.keycloak.org/documentation
- **MinIO**: https://min.io/docs/minio/linux/index.html
