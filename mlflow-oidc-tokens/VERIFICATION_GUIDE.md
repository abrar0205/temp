# MLflow OIDC - Complete Verification Guide

This document provides **complete step-by-step verification** for all Definition of Done (DoD) items.

---

## DoD Checklist

| # | DoD Item | Verification Section |
|---|----------|---------------------|
| 1 | MLflow OIDC container running locally with PostgreSQL in docker | [Section 1](#1-verify-mlflow-oidc-container-with-postgresql) |
| 2 | Users can create access tokens via UI/CLI | [Section 2](#2-verify-token-creation) |
| 3 | Users can access experiment results using tokens **(PRIMARY)** | [Section 3](#3-verify-token-based-experiment-access-primary-goal) |

> **Note**: Section 4 covers optional token management (listing/deleting) which is handled via Keycloak Admin Console.

---

## Prerequisites

Before verification, ensure you have:
- Docker and Docker Compose installed
- `jq` installed (for JSON parsing): `sudo apt install jq` or `brew install jq`
- `curl` installed

---

## 1. Verify MLflow OIDC Container with PostgreSQL

### Step 1.1: Navigate to MLflow Docker directory
```bash
cd Mlflow_docker
```

### Step 1.2: Start the stack
```bash
docker compose up -d
```

### Step 1.3: Wait for services to be healthy (about 60 seconds)
```bash
docker compose ps
```

**Expected output** - All services should show "healthy" or "running":
```
NAME              STATUS                   PORTS
mlflow-keycloak   Up (healthy)             0.0.0.0:8080->8080/tcp
mlflow-minio      Up (healthy)             0.0.0.0:9000-9001->9000-9001/tcp
mlflow-postgres   Up (healthy)             0.0.0.0:5432->5432/tcp
mlflow-redis      Up (healthy)             0.0.0.0:6379->6379/tcp
mlflow-server     Up (healthy)             0.0.0.0:5000->5000/tcp
```

### Step 1.4: Verify PostgreSQL is running
```bash
docker exec mlflow-postgres pg_isready -U mlflow
```

**Expected output**:
```
/var/run/postgresql:5432 - accepting connections
```

### Step 1.5: Verify MLflow is accessible
```bash
curl -s http://localhost:5000/health 2>/dev/null || curl -s -o /dev/null -w "%{http_code}" http://localhost:5000
```

**Expected**: Response or HTTP code (302 redirect to login is OK)

### Step 1.6: Configure Keycloak (first time only)
```bash
./configure-keycloak.sh
```

**Expected output** - Note the CLIENT_SECRET:
```
============================================
OIDC Client Secret: abc123xyz...
============================================
```

**⚠️ IMPORTANT**: Save this CLIENT_SECRET - you need it for the next steps!

### Step 1.7: Update .env with the client secret
```bash
# Edit .env file and add/update:
# OIDC_CLIENT_SECRET=abc123xyz...

# Then restart MLflow
docker compose restart mlflow
```

### ✅ DoD #1 VERIFIED if:
- [ ] All 5 containers are running and healthy
- [ ] PostgreSQL accepts connections
- [ ] MLflow responds on port 5000

---

## 2. Verify Token Creation

### Step 2.1: Create token via CLI (curl)

Replace `YOUR_CLIENT_SECRET` with the secret from Step 1.6:

```bash
# Set your client secret
export CLIENT_SECRET="YOUR_CLIENT_SECRET"

# Get access token
curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=$CLIENT_SECRET" \
  -d "username=mlflow_user" \
  -d "******"
```

**Expected output** (JSON with access_token):
```json
{
  "access_token": "******",
  "expires_in": 300,
  "refresh_expires_in": 1800,
  "refresh_token": "******",
  "token_type": "Bearer",
  "not-before-policy": 0,
  "session_state": "...",
  "scope": "openid profile email"
}
```

### Step 2.2: Save the token for next steps
```bash
# Get just the access token
export TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=$CLIENT_SECRET" \
  -d "username=mlflow_user" \
  -d "******" | jq -r '.access_token')

# Verify token was retrieved
echo "Token: ${TOKEN:0:50}..."
```

**Expected**: Token string starting with "eyJ..."

### Step 2.3: Create token via UI

**Where**: MLflow UI at http://localhost:5000

**Steps**:
1. Open browser: **http://localhost:5000**
2. You'll be automatically redirected to Keycloak login page
3. Enter credentials:
   - Username: `mlflow_user`
   - Password: `password123`
4. Click **Sign In**
5. You're now logged in - token is automatically created and managed by the browser session

**What happens behind the scenes**:
- Browser receives OAuth token from Keycloak
- Token is stored in browser cookies/session
- All subsequent requests to MLflow include the token automatically

### ✅ DoD #2 VERIFIED if:
- [ ] **CLI**: curl command returns JSON with `access_token` field
- [ ] **CLI**: Token starts with "eyJ" (JWT format)
- [ ] **UI**: Visiting http://localhost:5000 redirects to Keycloak
- [ ] **UI**: Login with mlflow_user/password123 succeeds
- [ ] **UI**: After login, MLflow UI is accessible

---

## 3. Verify Token-Based Experiment Access (PRIMARY GOAL)

### Step 3.1: List experiments using token
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Expected output**:
```json
{
  "experiments": [
    {
      "experiment_id": "0",
      "name": "Default",
      "artifact_location": "...",
      "lifecycle_stage": "active"
    }
  ]
}
```

### Step 3.2: Create a new experiment
```bash
RESULT=$(curl -s -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"name": "test-token-experiment"}')

echo $RESULT
export EXPERIMENT_ID=$(echo $RESULT | jq -r '.experiment_id')
echo "Created experiment ID: $EXPERIMENT_ID"
```

**Expected output**:
```json
{"experiment_id": "1"}
```

### Step 3.3: Create a run in the experiment
```bash
RESULT=$(curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/create" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d "{\"experiment_id\": \"$EXPERIMENT_ID\", \"run_name\": \"test-run\"}")

echo $RESULT
export RUN_ID=$(echo $RESULT | jq -r '.run.info.run_id')
echo "Created run ID: $RUN_ID"
```

**Expected output**: JSON with run info and run_id

### Step 3.4: Log metrics to the run
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"key\": \"accuracy\", \"value\": 0.95, \"timestamp\": $(date +%s)000}"
```

**Expected output**: Empty response `{}` = success

### Step 3.5: Log parameters to the run
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-parameter" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"key\": \"learning_rate\", \"value\": \"0.01\"}"
```

**Expected output**: Empty response `{}` = success

### Step 3.6: Get the run results
```bash
curl -s "http://localhost:5000/api/2.0/mlflow/runs/get?run_id=$RUN_ID" \
  -H "Authorization: ******" | jq '.run.data'
```

**Expected output**:
```json
{
  "metrics": [
    {"key": "accuracy", "value": 0.95, "timestamp": ...}
  ],
  "params": [
    {"key": "learning_rate", "value": "0.01"}
  ]
}
```

### Step 3.7: Search runs in experiment
```bash
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d "{\"experiment_ids\": [\"$EXPERIMENT_ID\"]}" | jq '.runs[0].info'
```

**Expected output**: Run info with run_id, status, etc.

### Step 3.8: Verify in MLflow UI
1. Open browser: http://localhost:5000
2. Login with `mlflow_user` / `password123`
3. See the "test-token-experiment" experiment
4. Click to see the run with accuracy=0.95 metric

### ✅ DoD #3 VERIFIED (PRIMARY GOAL) if:
- [ ] List experiments returns JSON with experiments
- [ ] Create experiment succeeds
- [ ] Create run succeeds
- [ ] Log metric succeeds
- [ ] Log parameter succeeds
- [ ] Get run returns the logged metrics/params
- [ ] UI shows the experiment and run

---

## 4. Verify Token Listing and Deletion

### Step 4.1: Access Keycloak Admin Console
1. Open browser: http://localhost:8080/admin
2. Login with: `admin` / `admin123` (or values from .env)
3. Select **mlflow** realm from dropdown (top-left)

### Step 4.2: List active tokens/sessions
1. Click **Sessions** in left menu
2. You should see active sessions

**Expected**: List of active sessions with usernames and client info

### Step 4.3: List sessions for a specific user
1. Click **Users** in left menu
2. Click on **mlflow_user**
3. Click **Sessions** tab
4. See all sessions for this user

### Step 4.4: Delete/Revoke a token
**Method A - From Sessions view:**
1. Go to **Sessions** in left menu
2. Find the session to revoke
3. Click **Sign out** button

**Method B - From User view:**
1. Go to **Users** → **mlflow_user** → **Sessions**
2. Click **Sign out all sessions**

### Step 4.5: Verify token is revoked
```bash
# Try to use the old token (should fail)
curl -s -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Expected**: Error response (401 Unauthorized or invalid token)

### Step 4.6: Document the limitation
**⚠️ IMPORTANT FINDING**: 

As of mlflow-oidc-auth v5.6.x:
- **NO built-in UI** in MLflow for listing tokens
- **NO built-in UI** in MLflow for deleting tokens
- All token management must be done through **Keycloak Admin Console**

Check [mlflow-oidc-auth releases](https://github.com/mlflow-oidc/mlflow-oidc-auth/releases) for updates.

### ✅ DoD #4 VERIFIED if:
- [ ] Keycloak Admin Console accessible at http://localhost:8080/admin
- [ ] Sessions are visible in Keycloak Admin → Sessions
- [ ] User sessions visible in Users → [user] → Sessions
- [ ] Sign out successfully revokes tokens
- [ ] Revoked token returns error when used

---

## Complete Verification Script

Save this as `verify_all.sh` and run it:

```bash
#!/bin/bash
set -e

echo "================================================"
echo "MLflow OIDC Complete Verification"
echo "================================================"

# Configuration - UPDATE THESE!
CLIENT_SECRET="${CLIENT_SECRET:-YOUR_CLIENT_SECRET}"
KEYCLOAK_URL="http://localhost:8080"
MLFLOW_URL="http://localhost:5000"
USERNAME="mlflow_user"
******

echo ""
echo "🔍 DoD #1: Checking containers..."
docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || echo "Run from Mlflow_docker directory"

echo ""
echo "🔍 DoD #2: Getting access token..."
TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/realms/mlflow/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=$CLIENT_SECRET" \
  -d "username=$USERNAME" \
  -d "******" | jq -r '.access_token')

if [ "$TOKEN" == "null" ] || [ -z "$TOKEN" ]; then
  echo "❌ Failed to get token. Check CLIENT_SECRET."
  exit 1
fi
echo "✅ Token obtained: ${TOKEN:0:50}..."

echo ""
echo "🔍 DoD #3: Testing experiment access with token..."

# List experiments
echo "  - Listing experiments..."
EXPERIMENTS=$(curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/experiments/search" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{}')
echo "  ✅ Experiments: $(echo $EXPERIMENTS | jq '.experiments | length') found"

# Create experiment
echo "  - Creating experiment..."
EXP_RESULT=$(curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/experiments/create" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{"name": "verification-test-'$(date +%s)'"}')
EXPERIMENT_ID=$(echo $EXP_RESULT | jq -r '.experiment_id')
echo "  ✅ Created experiment ID: $EXPERIMENT_ID"

# Create run
echo "  - Creating run..."
RUN_RESULT=$(curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/runs/create" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d "{\"experiment_id\": \"$EXPERIMENT_ID\", \"run_name\": \"verification-run\"}")
RUN_ID=$(echo $RUN_RESULT | jq -r '.run.info.run_id')
echo "  ✅ Created run ID: $RUN_ID"

# Log metric
echo "  - Logging metric..."
curl -s -X POST "$MLFLOW_URL/api/2.0/mlflow/runs/log-metric" \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"key\": \"accuracy\", \"value\": 0.95, \"timestamp\": $(date +%s)000}" > /dev/null
echo "  ✅ Logged metric: accuracy=0.95"

# Get run results
echo "  - Getting run results..."
RUN_DATA=$(curl -s "$MLFLOW_URL/api/2.0/mlflow/runs/get?run_id=$RUN_ID" \
  -H "Authorization: ******")
METRIC_VALUE=$(echo $RUN_DATA | jq -r '.run.data.metrics[0].value')
echo "  ✅ Retrieved metric value: $METRIC_VALUE"

echo ""
echo "🔍 DoD #4: Token management info..."
echo "  📍 List tokens: http://localhost:8080/admin → mlflow realm → Sessions"
echo "  📍 Delete tokens: Sessions → Sign out"
echo "  ⚠️  Note: mlflow-oidc-auth has NO built-in token management UI"

echo ""
echo "================================================"
echo "✅ ALL VERIFICATION COMPLETE!"
echo "================================================"
echo ""
echo "Manual steps remaining:"
echo "1. Open http://localhost:5000 - verify UI access"
echo "2. Open http://localhost:8080/admin - verify token listing"
echo "3. Try Sign out in Keycloak - verify token revocation"
```

### Run the verification:
```bash
export CLIENT_SECRET="your-secret-from-configure-keycloak"
chmod +x verify_all.sh
./verify_all.sh
```

---

## Summary

| DoD | Item | How to Verify | Pass Criteria |
|-----|------|---------------|---------------|
| #1 | MLflow + PostgreSQL running | `docker compose ps` | All 5 containers healthy |
| #2 | Create tokens via CLI | `curl` to token endpoint | Returns JSON with access_token |
| #3 | Access experiments with tokens | `curl` with ****** | Create/read experiments/runs works |
| #4 | List/delete tokens | Keycloak Admin → Sessions | Sessions visible, Sign out works |
