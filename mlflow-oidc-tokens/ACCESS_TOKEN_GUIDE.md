# MLflow OIDC Access Token Guide

## 🎯 Primary Goal: Access Experiment Results Using Tokens

This guide shows how users can **access and publish experiment results using access tokens** with the MLflow OIDC deployment.

> **Prerequisites**: MLflow OIDC stack running (see `Mlflow_docker/` on main branch)

---

## Step-by-Step: Access Experiments with Tokens

### Step 1: Get an Access Token

**Where**: Keycloak Token Endpoint

```bash
# Get access token from Keycloak
TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "username=mlflow_user" \
  -d "******" | jq -r '.access_token')

echo "Your token: $TOKEN"
```

> **Note**: Get `YOUR_CLIENT_SECRET` from the output of `./configure-keycloak.sh`

### Step 2: Access Experiment Results

**Where**: MLflow REST API at `http://localhost:5000/api/2.0/mlflow/`

#### List All Experiments
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### Get Runs from an Experiment
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["0"]}'
```

#### Get Specific Run Details
```bash
curl "http://localhost:5000/api/2.0/mlflow/runs/get?run_id=YOUR_RUN_ID" \
  -H "Authorization: ******"
```

#### Get Metrics for a Run
```bash
curl "http://localhost:5000/api/2.0/mlflow/metrics/get-history?run_id=YOUR_RUN_ID&metric_key=accuracy" \
  -H "Authorization: ******"
```

### Step 3: Publish Results (Create/Log Experiments)

#### Create New Experiment
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-new-experiment"}'
```

#### Start a Run
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/create" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": "1", "run_name": "my-run"}'
```

#### Log Metrics
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"run_id": "YOUR_RUN_ID", "key": "accuracy", "value": 0.95}'
```

#### Log Parameters
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-parameter" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"run_id": "YOUR_RUN_ID", "key": "learning_rate", "value": "0.01"}'
```

---

## Complete Example Script

Save this as `access_mlflow.sh`:

```bash
#!/bin/bash

# Configuration
KEYCLOAK_URL="http://localhost:8080"
MLFLOW_URL="http://localhost:5000"
CLIENT_SECRET="YOUR_CLIENT_SECRET"  # From configure-keycloak.sh
USERNAME="mlflow_user"
PASSWORD="password123"

# Step 1: Get access token
echo "Getting access token..."
TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/realms/mlflow/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=$CLIENT_SECRET" \
  -d "username=$USERNAME" \
  -d "******" | jq -r '.access_token')

if [ "$TOKEN" == "null" ] || [ -z "$TOKEN" ]; then
  echo "❌ Failed to get token"
  exit 1
fi
echo "✅ Token obtained"

# Step 2: List experiments
echo ""
echo "Listing experiments..."
curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/experiments/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{}' | jq '.experiments[] | {name, experiment_id}'

# Step 3: Get runs from default experiment
echo ""
echo "Getting runs from experiment 0..."
curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/runs/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["0"]}' | jq '.runs[] | {run_id: .info.run_id, status: .info.status}'

echo ""
echo "✅ Successfully accessed MLflow with token!"
```

---

## Token Management (Secondary)

### How to List Active Tokens
**Where**: Keycloak Admin Console → http://localhost:8080/admin

1. Login with admin credentials
2. Select **mlflow** realm
3. Click **Sessions** in left menu
4. View all active sessions/tokens

### How to Delete/Revoke Tokens
**Where**: Keycloak Admin Console → Sessions

1. Find the session
2. Click **Sign out**

> ⚠️ **Note**: mlflow-oidc-auth v5.6.x has **NO built-in UI** for listing/deleting tokens. Use Keycloak Admin Console.

### How to Configure Token Expiry
**Where**: Keycloak Admin Console → Realm Settings → Tokens

- **Access Token Lifespan**: Default 5 minutes
- **SSO Session Max**: Default 10 hours

---

## Quick Reference

| Task | URL | Method |
|------|-----|--------|
| Get Token | http://localhost:8080/realms/mlflow/protocol/openid-connect/token | POST with credentials |
| List Experiments | http://localhost:5000/api/2.0/mlflow/experiments/search | POST with Bearer token |
| Search Runs | http://localhost:5000/api/2.0/mlflow/runs/search | POST with Bearer token |
| Log Metrics | http://localhost:5000/api/2.0/mlflow/runs/log-metric | POST with Bearer token |
| Manage Tokens | http://localhost:8080/admin | Keycloak Admin UI |
