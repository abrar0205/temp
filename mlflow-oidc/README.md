# MLflow OIDC Local Deployment with PostgreSQL

This directory contains a complete Docker-based setup for running **MLflow OIDC v5.7.0** (build 20251227) with Keycloak for enterprise authentication and PostgreSQL as the tracking backend.

## 📋 Table of Contents

- [Overview](#overview)
- [Version Information](#version-information)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Keycloak Configuration](#keycloak-configuration)
- [Token Management](#token-management)
- [Accessing Experiments](#accessing-experiments)
- [Token Listing and Deletion](#token-listing-and-deletion)
- [Troubleshooting](#troubleshooting)

## Overview

This setup provides:
- **MLflow OIDC Server** (v5.7.0) with enterprise authentication
- **Keycloak** as the Identity Provider for OIDC authentication
- **PostgreSQL** as the tracking backend for storing experiment metadata
- **Token-based access** via OAuth2/OIDC tokens
- **Scripts** for managing users and accessing experiments

## Version Information

| Component | Version | Source |
|-----------|---------|--------|
| MLflow OIDC | v5.7.0 (build 20251227) | `ghcr.io/mlflow/mlflow-oidc-auth:v5.7.0` |
| Keycloak | 23.0 | `quay.io/keycloak/keycloak:23.0` |
| PostgreSQL | 15-alpine | `postgres:15-alpine` |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Docker Network (mlflow-network)                       │
│                                                                              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────────┐│
│  │                 │   │                 │   │                             ││
│  │   PostgreSQL    │   │   Keycloak      │   │   MLflow OIDC Server        ││
│  │   (MLflow DB)   │   │   (Port 8080)   │   │   (Port 5000)               ││
│  │   Port 5432     │   │                 │   │                             ││
│  │                 │   │  ┌───────────┐  │   │  ┌─────────────────────┐   ││
│  │ - experiments   │   │  │ Admin     │  │◄──┼──│  OIDC Authentication│   ││
│  │ - runs          │   │  │ Console   │  │   │  │  (v5.7.0)           │   ││
│  │ - metrics       │   │  └───────────┘  │   │  └─────────────────────┘   ││
│  │                 │   │                 │   │                             ││
│  └────────┬────────┘   │  ┌───────────┐  │   │  ┌─────────────────────┐   ││
│           │            │  │ mlflow    │  │   │  │  Experiment         │   ││
│           └────────────┼──│ realm     │  │   │  │  Tracking           │   ││
│                        │  └───────────┘  │   │  └─────────────────────┘   ││
│                        └─────────────────┘   └─────────────────────────────┘│
│                                                          │                   │
│  ┌─────────────────┐                                     │                   │
│  │   PostgreSQL    │                                     │                   │
│  │  (Keycloak DB)  │                                     │                   │
│  └─────────────────┘                                     │                   │
└──────────────────────────────────────────────────────────│───────────────────┘
                                                           │
                                                           ▼
                                                    ┌──────────────┐
                                                    │   Clients    │
                                                    │  (UI/CLI/    │
                                                    │   Python)    │
                                                    └──────────────┘
```

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ (for running scripts)

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to set your credentials
```

### 3. Start the Services

```bash
# Start all services (MLflow OIDC, Keycloak, PostgreSQL)
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f mlflow

# Wait for Keycloak to be ready (may take 1-2 minutes)
docker compose logs -f keycloak
```

### 4. Configure Keycloak (First Time Setup)

1. **Access Keycloak Admin Console**: http://localhost:8080
2. **Login with admin credentials**: `admin` / `admin`
3. **Create a new realm**:
   - Click "Create Realm"
   - Name: `mlflow`
   - Click "Create"

4. **Create a client for MLflow**:
   - Go to Clients → Create client
   - Client ID: `mlflow`
   - Client Protocol: `openid-connect`
   - Click "Next"
   - Enable "Client authentication"
   - Click "Save"

5. **Configure client settings**:
   - Root URL: `http://localhost:5000`
   - Valid redirect URIs: `http://localhost:5000/*`
   - Web origins: `http://localhost:5000`
   - Click "Save"

6. **Get client secret**:
   - Go to Credentials tab
   - Copy the "Client secret"
   - Update your `.env` file with this secret

7. **Create a test user**:
   - Go to Users → Add user
   - Username: `testuser`
   - Email: `testuser@example.com`
   - Click "Create"
   - Go to Credentials tab → Set password

### 5. Access MLflow UI

Open your browser and navigate to: **http://localhost:5000**

You will be redirected to Keycloak for authentication.

### 6. Verify Token-Based Access

```bash
# Install requirements
pip install requests mlflow

# Get an access token from Keycloak
ACCESS_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "username=testuser" \
  -d "password=testpassword" | jq -r '.access_token')

# Use token to access MLflow API
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Keycloak Configuration

### Realm Configuration

The MLflow OIDC integration expects a Keycloak realm with:
- Realm name: `mlflow` (configurable via `OIDC_DISCOVERY_URL`)
- Client ID: `mlflow` (configurable via `OIDC_CLIENT_ID`)
- Client authentication: Enabled (confidential client)

### User Management

All user management is done through Keycloak Admin Console:

1. **Create Users**: Users → Add user
2. **Assign Roles**: Users → Select user → Role mappings
3. **Reset Passwords**: Users → Select user → Credentials
4. **Delete Users**: Users → Select user → Delete

## Token Management

### Getting Access Tokens

**Via Keycloak Token Endpoint (Password Grant):**
```bash
curl -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "username=testuser" \
  -d "password=testpassword"
```

**Via Python:**
```python
import requests

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

token = get_access_token("testuser", "testpassword", "YOUR_CLIENT_SECRET")
print(f"Access Token: {token}")
```

### Using Access Tokens

**With MLflow Python SDK:**
```python
import os
import mlflow

# Set the access token
os.environ["MLFLOW_TRACKING_TOKEN"] = "YOUR_ACCESS_TOKEN"
mlflow.set_tracking_uri("http://localhost:5000")

# Now all MLflow operations use the token
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
```

**With REST API:**
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Token Listing and Deletion

### ✅ Full Token Management via Keycloak

With MLflow OIDC v5.7.0 using Keycloak, you have **full token management capabilities** through the Keycloak Admin Console.

### Listing Active Sessions/Tokens

**Via Keycloak Admin Console:**
1. Go to http://localhost:8080/admin
2. Select `mlflow` realm
3. Navigate to **Sessions** in the left menu
4. View all active sessions across all users

**Via Keycloak Admin API:**
```bash
# Get admin token first
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin" \
  -d "password=admin" \
  -d "grant_type=password" \
  -d "client_id=admin-cli" | jq -r '.access_token')

# List all sessions for a specific user
USER_ID="user-uuid-here"
curl -s "http://localhost:8080/admin/realms/mlflow/users/$USER_ID/sessions" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# List all client sessions
curl -s "http://localhost:8080/admin/realms/mlflow/clients/CLIENT_UUID/user-sessions" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Deleting/Revoking Tokens

**Via Keycloak Admin Console:**
1. Go to Sessions
2. Find the session to revoke
3. Click "Sign out" or "Logout all"

**Via Keycloak Admin API:**
```bash
# Logout a specific user's all sessions
curl -X POST "http://localhost:8080/admin/realms/mlflow/users/$USER_ID/logout" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Revoke a specific session
curl -X DELETE "http://localhost:8080/admin/realms/mlflow/sessions/$SESSION_ID" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Via Token Revocation Endpoint:**
```bash
# Revoke a specific token
curl -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/revoke" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "token=TOKEN_TO_REVOKE"
```

### Token Management Capabilities with OIDC

| Feature | MLflow OIDC + Keycloak | How to Access |
|---------|------------------------|---------------|
| Create Token | ✅ | Token endpoint / Login |
| Use Token | ✅ | Authorization: Bearer header |
| List Tokens | ✅ | Keycloak Admin Console / API |
| Delete Token | ✅ | Keycloak Admin Console / API |
| Rotate Token | ✅ | Refresh token endpoint |
| Token Expiry | ✅ | Configurable in Keycloak |
| Token Scopes | ✅ | Configurable per client |

### Configuring Token Expiration

In Keycloak Admin Console:
1. Go to Realm Settings → Tokens
2. Configure:
   - Access Token Lifespan (default: 5 minutes)
   - Access Token Lifespan For Implicit Flow
   - Client login timeout
   - Refresh Token Max Reuse

## Accessing Experiments

### Via Python SDK with OAuth Token

```python
import os
import mlflow
import requests

# Get token from Keycloak
def get_token():
    response = requests.post(
        "http://localhost:8080/realms/mlflow/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": "mlflow",
            "client_secret": "YOUR_CLIENT_SECRET",
            "username": "testuser",
            "password": "testpassword",
        }
    )
    return response.json()["access_token"]

# Configure MLflow
os.environ["MLFLOW_TRACKING_TOKEN"] = get_token()
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
experiment_id = mlflow.create_experiment("my-oidc-experiment")

# Log a run
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

# Search experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
```

### Via REST API

```bash
# Get access token
ACCESS_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "username=testuser" \
  -d "password=testpassword" | jq -r '.access_token')

# List experiments
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'

# Create experiment
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-experiment"}'

# Log metrics
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"run_id": "RUN_ID", "key": "accuracy", "value": 0.95}'
```

## Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v
```

## Troubleshooting

### Cannot connect to MLflow

```bash
# Check if services are running
docker compose ps

# Check MLflow logs
docker compose logs mlflow

# Check Keycloak logs
docker compose logs keycloak

# Check PostgreSQL logs
docker compose logs postgres
```

### Keycloak not ready

Keycloak may take 1-2 minutes to start. Check the logs:

```bash
docker compose logs -f keycloak
```

Wait for: `Running the server in development mode`

### OIDC Authentication errors

```bash
# Verify Keycloak realm exists
curl -s http://localhost:8080/realms/mlflow/.well-known/openid-configuration

# Test token endpoint
curl -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_SECRET" \
  -d "username=testuser" \
  -d "password=testpassword"
```

### Database connection issues

```bash
# Check PostgreSQL is healthy
docker exec -it mlflow-postgres pg_isready -U mlflow

# Check Keycloak PostgreSQL
docker exec -it keycloak-postgres pg_isready -U keycloak
```

## Files in This Directory

```
mlflow-oidc/
├── docker-compose.yml      # Main Docker Compose configuration (MLflow OIDC v5.7.0)
├── auth_config.ini         # MLflow authentication configuration (for basic-auth fallback)
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── README.md               # This documentation
├── oidc-setup/             # Alternative standalone OIDC setup
│   ├── docker-compose.yml
│   └── README.md
└── scripts/
    ├── requirements.txt
    ├── token_access_example.py  # Example script for token-based access
    └── token_manager.py         # Token/user management utilities
```
