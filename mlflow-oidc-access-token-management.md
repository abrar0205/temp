# MLflow OIDC Access Token Management Guide

Complete guide for managing access tokens in MLflow with OIDC authentication - listing, deleting, and UI verification.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [OIDC Login Procedure](#oidc-login-procedure)
3. [Token Listing Methods](#token-listing-methods)
4. [Token Deletion Methods](#token-deletion-methods)
5. [UI Support for Token Management](#ui-support-for-token-management)
6. [Container Cleanup](#container-cleanup)

---

## Quick Reference

### Token Operations Summary

| Operation | CLI Command | API Endpoint | UI Support |
|-----------|-------------|--------------|------------|
| **Login** | `mlflow login` | POST `/api/2.0/mlflow/users/login` | ✅ Yes |
| **List Tokens** | `curl -u user:pass .../tokens` | GET `/api/2.0/mlflow/tokens` | ⚠️ Limited |
| **Delete Token** | `curl -X DELETE .../tokens/{id}` | DELETE `/api/2.0/mlflow/tokens/{id}` | ⚠️ Limited |
| **Create Token** | Via API | POST `/api/2.0/mlflow/tokens` | ✅ Yes |

---

## OIDC Login Procedure

### Method 1: Web UI Login

1. **Open MLflow URL**: Navigate to `http://localhost:5000`
2. **Enter Credentials**: When prompted, enter username and password
3. **Successful Login**: You'll see the MLflow experiments dashboard

### Method 2: CLI Login

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Login with credentials
mlflow login

# Or use environment variables
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password
```

### Method 3: API Login (Token-Based)

```bash
# Use basic auth with curl
curl -u admin:password http://localhost:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{}'

# Or use the Authorization header
curl http://localhost:5000/api/2.0/mlflow/experiments/search \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Method 4: Python SDK Login

```python
import mlflow
import os

# Set credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Verify connection
client = mlflow.MlflowClient()
experiments = client.search_experiments()
print(f"Connected! Found {len(experiments)} experiments")
```

---

## Token Listing Methods

### Method 1: CLI with curl

```bash
# List all tokens for current user
curl -u admin:password \
  "http://localhost:5000/api/2.0/mlflow/tokens" \
  -H "Content-Type: application/json"
```

**Expected Response:**
```json
{
  "tokens": [
    {
      "token_id": "tok_abc123",
      "name": "my-token",
      "created_at": "2024-01-15T10:30:00Z",
      "expires_at": "2024-07-15T10:30:00Z"
    }
  ]
}
```

### Method 2: Python SDK

```python
import requests
from requests.auth import HTTPBasicAuth

MLFLOW_URL = "http://localhost:5000"
USERNAME = "admin"
PASSWORD = "password"

def list_tokens():
    """List all access tokens"""
    response = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/tokens",
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )
    
    if response.status_code == 200:
        tokens = response.json().get("tokens", [])
        print(f"Found {len(tokens)} token(s):")
        for token in tokens:
            print(f"  - ID: {token['token_id']}")
            print(f"    Name: {token['name']}")
            print(f"    Created: {token['created_at']}")
        return tokens
    else:
        print(f"Error: {response.status_code}")
        return None

# Run it
tokens = list_tokens()
```

### Method 3: UI-Based Listing

1. Log in to MLflow UI at `http://localhost:5000`
2. Click on your username (top-right corner)
3. Select "Access Tokens" or "Settings"
4. View your token list

**Note:** UI token management depends on MLflow version. See [UI Support](#ui-support-for-token-management).

---

## Token Deletion Methods

### Method 1: CLI with curl

```bash
# Delete a specific token by ID
TOKEN_ID="tok_abc123"

curl -X DELETE \
  "http://localhost:5000/api/2.0/mlflow/tokens/${TOKEN_ID}" \
  -u admin:password \
  -H "Content-Type: application/json"
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Token deleted"
}
```

### Method 2: Python SDK

```python
import requests
from requests.auth import HTTPBasicAuth

MLFLOW_URL = "http://localhost:5000"
USERNAME = "admin"
PASSWORD = "password"

def delete_token(token_id):
    """Delete a token by ID"""
    response = requests.delete(
        f"{MLFLOW_URL}/api/2.0/mlflow/tokens/{token_id}",
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )
    
    if response.status_code == 200:
        print(f"Successfully deleted token: {token_id}")
        return True
    elif response.status_code == 404:
        print(f"Token not found: {token_id}")
        return False
    else:
        print(f"Error: {response.status_code}")
        return False

# Delete a token
delete_token("tok_abc123")
```

### Method 3: UI-Based Deletion

1. Navigate to Access Tokens (see listing section)
2. Find the token to delete
3. Click the delete/revoke button (🗑️)
4. Confirm deletion

---

## UI Support for Token Management

### MLflow Version Comparison

| Feature | MLflow 2.5-2.8 | MLflow 2.9-2.12 | MLflow 2.13+ |
|---------|----------------|-----------------|--------------|
| UI Login | ✅ | ✅ | ✅ |
| Token Creation (UI) | ❌ | Partial | ✅ |
| Token Listing (UI) | ❌ | ❌ | ✅ |
| Token Deletion (UI) | ❌ | ❌ | ✅ |
| Token Refresh | ✅ Auto | ✅ Auto | ✅ Auto |

### Check Your MLflow Version

```bash
# Check MLflow version
mlflow --version

# Check installed plugins
pip list | grep mlflow
```

### MLflow OIDC Plugin Support

The `mlflow-oidc-auth` plugin provides:

| Feature | Support |
|---------|---------|
| OIDC Login | ✅ Full |
| Token via API | ✅ Full |
| Token via UI | ⚠️ Version dependent |

**Recommendation:** Always use CLI/API for reliable token management.

---

## Container Cleanup

### Stop Services

```bash
# Stop all containers (keep data)
docker compose stop

# Or stop specific container
docker stop mlflow-server
```

### Remove Containers

```bash
# Stop and remove containers (keep data volumes)
docker compose down

# Stop, remove containers AND delete all data
docker compose down -v

# Full cleanup including images
docker compose down -v --rmi local
```

### Manual Cleanup

```bash
# List MLflow containers
docker ps -a | grep mlflow

# Stop all MLflow containers
docker stop mlflow-server mlflow-postgres mlflow-redis mlflow-minio mlflow-keycloak

# Remove all MLflow containers
docker rm mlflow-server mlflow-postgres mlflow-redis mlflow-minio mlflow-keycloak

# Remove volumes
docker volume rm mlflow-postgres-data mlflow-minio-data

# Remove network
docker network rm mlflow-network
```

### Cleanup Script

Save this as `cleanup.sh`:

```bash
#!/bin/bash
echo "Stopping MLflow services..."
docker compose down -v

echo "Removing dangling resources..."
docker system prune -f

echo "Cleanup complete!"
```

Run with:
```bash
chmod +x cleanup.sh
./cleanup.sh
```

### Kubernetes Cleanup

```bash
# Delete deployments
kubectl delete deployment mlflow-server keycloak -n mlflow

# Delete services
kubectl delete service mlflow-server keycloak -n mlflow

# Delete secrets and configmaps
kubectl delete secret mlflow-oidc-secret -n mlflow
kubectl delete configmap mlflow-config -n mlflow

# Delete PVCs
kubectl delete pvc mlflow-pvc keycloak-pvc -n mlflow

# Or delete entire namespace
kubectl delete namespace mlflow
```

---

## Verification Commands Cheat Sheet

```bash
# 1. Check all containers are running
docker ps --format "table {{.Names}}\t{{.Status}}"

# 2. Check MLflow health
curl http://localhost:5000/health

# 3. Test token authentication
curl -u admin:password http://localhost:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" -d '{}'

# 4. Create experiment with token
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -u admin:password -H "Content-Type: application/json" \
  -d '{"name": "test-experiment"}'

# 5. Access experiment results with token
curl -u admin:password "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["1"]}'

# 6. Verify PostgreSQL storage
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT * FROM experiments;"
```

---

## Summary

### DoD Checklist for Token Management

| Requirement | How to Verify |
|-------------|---------------|
| ✅ OIDC login documented | See [Login Procedure](#oidc-login-procedure) |
| ✅ Token listing (CLI/API/UI) | See [Token Listing](#token-listing-methods) |
| ✅ Token deletion (CLI/API/UI) | See [Token Deletion](#token-deletion-methods) |
| ✅ UI support research | See [UI Support](#ui-support-for-token-management) |
| ✅ Container cleanup | See [Container Cleanup](#container-cleanup) |

---

*Last Updated: January 2025*
