# MLflow with Keycloak OIDC Authentication

MLflow deployment with Keycloak SSO authentication, Redis session storage, and MinIO artifact storage.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Components](#components)
- [Building the Docker Image](#building-the-docker-image)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [User Groups](#user-groups)
- [Management](#management)
- [Token Management](#token-management)
- [Troubleshooting](#troubleshooting)
- [Files](#files)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Docker Network (mlflow-network)                        │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   PostgreSQL    │  │     Redis       │  │     MinIO       │                 │
│  │   (Port 5432)   │  │   (Port 6379)   │  │  (9000/9001)    │                 │
│  │                 │  │                 │  │                 │                 │
│  │ ┌─────────────┐ │  │  Session        │  │  S3-compatible  │                 │
│  │ │ mlflow DB   │ │  │  Storage        │  │  Artifact       │                 │
│  │ ├─────────────┤ │  │                 │  │  Storage        │                 │
│  │ │ keycloak DB │ │  │                 │  │                 │                 │
│  │ └─────────────┘ │  └────────┬────────┘  └────────┬────────┘                 │
│  └────────┬────────┘           │                    │                          │
│           │                    │                    │                          │
│           ▼                    ▼                    ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │                      MLflow Server (Port 5000)                   │           │
│  │                                                                  │           │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │           │
│  │  │ OIDC Auth    │  │ Tracking API │  │ Artifact Storage     │  │           │
│  │  │ (Keycloak)   │  │              │  │ (MinIO/S3)           │  │           │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                    ▲                                            │
│                                    │                                            │
│  ┌─────────────────────────────────┴───────────────────────────────┐           │
│  │                      Keycloak (Port 8080)                        │           │
│  │                                                                  │           │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │           │
│  │  │ Admin        │  │ MLflow       │  │ Users & Groups       │  │           │
│  │  │ Console      │  │ Realm        │  │ Management           │  │           │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │           │
│  └─────────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Description | Port |
|-----------|-------------|------|
| **MLflow Server** | Tracking server with OIDC authentication | 5000 |
| **Keycloak** | OpenID Connect provider for SSO | 8080 |
| **PostgreSQL** | Backend storage for MLflow and Keycloak | 5432 |
| **Redis** | Session storage for MLflow OIDC | 6379 |
| **MinIO** | S3-compatible artifact storage | 9000 (API), 9001 (Console) |

## Building the Docker Image

The MLflow OIDC server image is built from the official [mlflow-tracking-server-docker](https://github.com/mlflow-oidc/mlflow-tracking-server-docker).

> **Note**: By default, this setup uses `ghcr.io/mlflow-oidc/mlflow-oidc-server:latest`. If you want to build your own image, follow the instructions below and update `MLFLOW_IMAGE` in your `.env` file.

### Build Command (Optional)

```bash
git clone https://github.com/mlflow-oidc/mlflow-tracking-server-docker.git
cd mlflow-tracking-server-docker
docker build --build-arg UV_HTTP_TIMEOUT=300 -t my-mlflow-oidc-server:latest .

# Then update .env:
# MLFLOW_IMAGE=my-mlflow-oidc-server:latest
```

### Command Explanation

| Part | Description |
|------|-------------|
| `docker build` | Build a Docker image from a Dockerfile |
| `--build-arg UV_HTTP_TIMEOUT=300` | Set timeout for uv package manager to 300 seconds (prevents build failures with large packages) |
| `-t my-mlflow-oidc-server:latest` | Tag the image (use your own registry path if pushing) |
| `.` | Build context (current directory) |

### What Gets Installed

The Dockerfile installs:
- Python 3.12
- MLflow 3.6.0
- mlflow-oidc-auth 5.6.1 (OIDC authentication plugin)
- psycopg2 (PostgreSQL driver)
- boto3 (S3/MinIO client)
- redis (Redis client for sessions)

### Push to Registry

```bash
# Login to your container registry
docker login registry.gitlab.com

# Push the image
docker push registry.gitlab.com/your-org/mlflow-oidc-server:latest
```

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
# Wait for all services to be healthy (takes ~60 seconds)
docker compose ps

# Run configuration script to set up realm, client, and test user
./configure-keycloak.sh
```

The script will output the `OIDC_CLIENT_SECRET`. Update your `.env` file with this value and restart MLflow:

```bash
# Update OIDC_CLIENT_SECRET in .env file
# Then restart MLflow
docker compose restart mlflow
```

### 4. Access MLflow

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| Keycloak Admin | http://localhost:8080 |
| MinIO Console | http://localhost:9001 |

**Default test user** (created by configure-keycloak.sh):
- Username: `mlflow_user`
- Password: `password123`

## Configuration

### Environment Variables

See `.env.example` for all options. Required variables:

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | Session encryption key (32+ random characters) |
| `OIDC_CLIENT_SECRET` | From Keycloak after running configure-keycloak.sh |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `KEYCLOAK_ADMIN_PASSWORD` | Keycloak admin password |
| `REDIS_PASSWORD` | Redis password |
| `MINIO_ROOT_PASSWORD` | MinIO root password |

### User Groups

Users must belong to one of these groups (configured in Keycloak):

| Group | Access Level |
|-------|--------------|
| `mlflow_admins` | Full administrative access |
| `mlflow_users` | Regular user access |

Update group names via `OIDC_ADMIN_GROUP_NAME` and `OIDC_GROUP_NAME` in `.env`.

## Management

### View Logs

```bash
docker compose logs -f mlflow
docker compose logs -f keycloak
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

## Token Management

### Overview

With Keycloak as the OIDC provider, you have full token management capabilities:

| Feature | Capability | How to Access |
|---------|------------|---------------|
| Create Token | ✅ | Token endpoint / Login |
| Use Token | ✅ | `Authorization: Bearer <token>` |
| List Tokens | ✅ | Keycloak Admin Console / API |
| Delete Token | ✅ | Keycloak Admin Console / API |
| Rotate Token | ✅ | Refresh token endpoint |
| Token Expiry | ✅ | Configurable in Keycloak |

### Listing Active Sessions/Tokens

**Via Keycloak Admin Console:**
1. Go to http://localhost:8080/admin
2. Select `mlflow` realm
3. Navigate to **Sessions** in the left menu
4. View all active sessions

**Via Keycloak Admin API:**
```bash
# Get admin token
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin" \
  -d "password=YOUR_ADMIN_PASSWORD" \
  -d "grant_type=password" \
  -d "client_id=admin-cli" | jq -r '.access_token')

# List all sessions for a user
curl -s "http://localhost:8080/admin/realms/mlflow/users/{user-id}/sessions" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Deleting/Revoking Tokens

**Via Admin Console:**
1. Go to Sessions
2. Find the session to revoke
3. Click "Sign out" or "Logout all"

**Via Admin API:**
```bash
# Logout all sessions for a user
curl -X POST "http://localhost:8080/admin/realms/mlflow/users/{user-id}/logout" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Getting Access Tokens Programmatically

```bash
# Get access token
ACCESS_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "username=mlflow_user" \
  -d "password=password123" | jq -r '.access_token')

# Use token with MLflow API
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Using Tokens with Python

```python
import os
import requests
import mlflow

def get_access_token(username, password, client_secret):
    response = requests.post(
        "http://localhost:8080/realms/mlflow/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": "mlflow",
            "client_secret": client_secret,
            "username": username,
            "password": password,
        }
    )
    return response.json()["access_token"]

# Get token
token = get_access_token("mlflow_user", "password123", "YOUR_CLIENT_SECRET")

# Configure MLflow
os.environ["MLFLOW_TRACKING_TOKEN"] = token
mlflow.set_tracking_uri("http://localhost:5000")

# Use MLflow
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
```

## Troubleshooting

### "Invalid scopes" error

- Ensure `.env` has `OIDC_SCOPE` (singular, not SCOPES)
- Value should be: `"openid profile email"`

### "Session error: Missing OAuth state"

```bash
# Check Redis is running
docker compose ps redis

# Verify Redis connection
docker exec mlflow-redis redis-cli -a YOUR_REDIS_PASSWORD ping
```

### "Authorization error: User is not allowed to login"

- User must be in `mlflow_admins` or `mlflow_users` group in Keycloak
- Check group membership in Keycloak admin console
- Verify `OIDC_ADMIN_GROUP_NAME` and `OIDC_GROUP_NAME` in `.env` match Keycloak groups

### "Internal Server Error"

```bash
# Check MLflow logs
docker compose logs mlflow

# Verify all environment variables are set correctly in .env
```

### Keycloak not starting

```bash
# Check Keycloak logs
docker compose logs keycloak

# Verify PostgreSQL is healthy
docker compose ps postgres
```

## Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Service definitions |
| `.env` | Environment configuration (DO NOT COMMIT) |
| `.env.example` | Template for `.env` file |
| `init-db.sh` | PostgreSQL initialization script |
| `configure-keycloak.sh` | Keycloak setup automation script |
| `.gitignore` | Excludes sensitive files from git |

## Support

For issues with:
- **MLflow OIDC**: https://github.com/mlflow-oidc/mlflow-oidc-auth
- **Keycloak**: https://www.keycloak.org/documentation
- **MinIO**: https://min.io/docs/minio/linux/index.html
