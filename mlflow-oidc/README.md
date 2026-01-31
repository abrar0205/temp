# MLflow OIDC Local Deployment with PostgreSQL

This directory contains a complete Docker-based setup for running MLflow with authentication and PostgreSQL as the tracking backend.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Authentication Options](#authentication-options)
- [Token Management](#token-management)
- [Accessing Experiments](#accessing-experiments)
- [Token Listing and Deletion](#token-listing-and-deletion)
- [Troubleshooting](#troubleshooting)

## Overview

This setup provides:
- **MLflow Server** with authentication enabled
- **PostgreSQL** as the tracking backend for storing experiment metadata
- **Token-based access** for API authentication
- **Scripts** for managing users and tokens

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Network (mlflow-network)               │
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │                     │      │                             │   │
│  │   PostgreSQL        │◄────►│   MLflow Server             │   │
│  │   (Port 5432)       │      │   (Port 5000)               │   │
│  │                     │      │                             │   │
│  │   - mlflow database │      │   - basic-auth enabled      │   │
│  │   - experiment data │      │   - artifact storage        │   │
│  │   - run metadata    │      │   - REST API                │   │
│  │                     │      │                             │   │
│  └─────────────────────┘      └─────────────────────────────┘   │
│                                          │                       │
└──────────────────────────────────────────│───────────────────────┘
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

# Edit .env to set your credentials (optional)
# Default credentials: admin / admin123
```

### 3. Start the Services

```bash
# Start MLflow and PostgreSQL
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f mlflow
```

### 4. Access MLflow UI

Open your browser and navigate to: **http://localhost:5000**

Login with:
- Username: `admin`
- Password: `admin123`

### 5. Verify Token-Based Access

```bash
# Install requirements
pip install requests mlflow

# Run the verification script
python scripts/token_access_example.py
```

## Authentication Options

### Option 1: MLflow Basic Auth (Default in this setup)

Uses username/password stored in a SQLite database.

**Credentials serve as tokens** - there are no separate API tokens.

```python
import mlflow

# Set tracking URI with credentials
mlflow.set_tracking_uri("http://localhost:5000")

# Set credentials as environment variables
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "admin123"

# Or use HTTP Basic Auth in API calls
import requests
from requests.auth import HTTPBasicAuth

response = requests.get(
    "http://localhost:5000/api/2.0/mlflow/experiments/search",
    auth=HTTPBasicAuth("admin", "admin123")
)
```

### Option 2: MLflow OIDC Auth (For Production)

For full OIDC integration with an Identity Provider (Keycloak, Okta, etc.), use the `mlflow-oidc-auth` plugin:

- GitHub: https://github.com/mlflow/mlflow-oidc-auth
- Provides OAuth2/OIDC integration
- Tokens managed by Identity Provider

## Token Management

### Creating Users (Tokens)

In MLflow basic-auth, users ARE tokens. Create a user to create an access credential:

**Via API:**
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/users/create" \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{"username": "newuser", "password": "newpassword123"}'
```

**Via Python Script:**
```bash
python scripts/token_manager.py create-user --username newuser --password newpassword123
```

### Using Tokens (Credentials)

**Python SDK:**
```python
import os
import mlflow

os.environ["MLFLOW_TRACKING_USERNAME"] = "newuser"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "newpassword123"
mlflow.set_tracking_uri("http://localhost:5000")

# Now all MLflow operations use these credentials
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
```

**REST API:**
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Content-Type: application/json" \
  -u newuser:newpassword123 \
  -d '{}'
```

## Token Listing and Deletion

### ⚠️ Important Note

**MLflow basic-auth does NOT provide UI or API for listing/deleting tokens.**

As of MLflow 2.10.x:
- No UI to list all tokens/users (except current user)
- No UI to delete tokens
- No REST API to list all tokens
- User deletion requires admin API access

### Workarounds

#### 1. Direct Database Access

Access the SQLite database directly to list users:

```bash
# If running locally
sqlite3 basic_auth.db "SELECT username, is_admin FROM users;"

# If running in Docker
docker exec -it mlflow-server sqlite3 /mlflow/basic_auth.db "SELECT username, is_admin FROM users;"
```

#### 2. Using the Token Manager Script

```bash
# List all users from database
python scripts/token_manager.py list-users --db-path ./basic_auth.db

# Delete a user (revoke their token)
python scripts/token_manager.py delete-user --username olduser

# Show detailed token management info
python scripts/token_manager.py info
```

#### 3. Delete Users via API

```bash
# Delete a user
curl -X DELETE "http://localhost:5000/api/2.0/mlflow/users/delete" \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{"username": "usertoremove"}'
```

#### 4. Change Password (Rotate Token)

```bash
curl -X PATCH "http://localhost:5000/api/2.0/mlflow/users/update-password" \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{"username": "existinguser", "password": "newpassword"}'
```

### Token Management Capabilities Comparison

| Feature | Basic Auth | OIDC Auth | How to Achieve |
|---------|-----------|-----------|----------------|
| Create Token | ✅ Create User | ✅ IdP Login | API/CLI |
| Use Token | ✅ HTTP Basic | ✅ Bearer Token | Auth Header |
| List Tokens | ❌ No UI/API | ✅ IdP Console | Direct DB query |
| Delete Token | ❌ No UI | ✅ IdP Console | API (delete user) |
| Rotate Token | ✅ Change Password | ✅ IdP Refresh | API |
| Token Expiry | ❌ Never expires | ✅ Configurable | N/A |

### Future Improvements (mlflow-oidc-auth)

The `mlflow-oidc-auth` project is actively developing better token management:

Check the latest version for updates:
```bash
# Check latest releases
curl -s https://api.github.com/repos/mlflow/mlflow-oidc-auth/releases/latest | grep tag_name
```

Features in development/planned:
- Token listing UI
- Token revocation UI
- Token expiration settings
- Audit logging for token usage

## Accessing Experiments

### Via Python SDK

```python
import os
import mlflow

# Configure authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "admin123"
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
experiment_id = mlflow.create_experiment("my-experiment")

# Log a run
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

# Search experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")

# Get runs from experiment
runs = mlflow.search_runs(experiment_ids=[experiment_id])
print(runs)
```

### Via REST API

```bash
# List experiments
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{}'

# Get specific experiment
curl -X GET "http://localhost:5000/api/2.0/mlflow/experiments/get?experiment_id=1" \
  -u admin:admin123

# Search runs in experiment
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{"experiment_ids": ["1"], "max_results": 100}'
```

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Troubleshooting

### Cannot connect to MLflow

```bash
# Check if services are running
docker-compose ps

# Check MLflow logs
docker-compose logs mlflow

# Check PostgreSQL logs
docker-compose logs postgres
```

### Authentication errors

```bash
# Verify credentials work
curl -v -u admin:admin123 http://localhost:5000/api/2.0/mlflow/experiments/search -d '{}'

# Check auth database
docker exec -it mlflow-server sqlite3 /mlflow/basic_auth.db "SELECT * FROM users;"
```

### Database connection issues

```bash
# Check PostgreSQL is healthy
docker exec -it mlflow-postgres pg_isready -U mlflow

# Verify connection from MLflow container
docker exec -it mlflow-server psql postgresql://mlflow:mlflow@postgres:5432/mlflow -c '\dt'
```

## Files in This Directory

```
mlflow-oidc/
├── docker-compose.yml      # Docker Compose configuration
├── auth_config.ini         # MLflow authentication configuration
├── .env.example            # Example environment variables
├── README.md               # This documentation
└── scripts/
    ├── token_access_example.py  # Example script for token-based access
    └── token_manager.py         # Token/user management utilities
```
