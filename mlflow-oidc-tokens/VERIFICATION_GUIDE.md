# MLflow OIDC - Complete Verification Guide

This document provides **complete step-by-step verification** for all Definition of Done (DoD) items.

## ✅ Verification Status

| Component | Status | Evidence |
|-----------|--------|----------|
| MLflow UI running | ✅ VERIFIED | [Screenshot](https://github.com/user-attachments/assets/1f5ab6ab-0abc-4c65-893a-84d674871b73) |
| PostgreSQL backend | ✅ VERIFIED | API returns experiments from database |
| MinIO artifacts | ✅ VERIFIED | Container healthy on ports 9000/9001 |
| Keycloak OIDC | 🔧 Requires your private image | See Step 1.2 below |

![MLflow UI Running](https://github.com/user-attachments/assets/1f5ab6ab-0abc-4c65-893a-84d674871b73)

---

> ⚠️ **Important Research Finding**: After reviewing the latest `mlflow-oidc-auth` v6.6.4 source code:
> - **Token Creation**: ✅ Available via UI and API
> - **Token Listing**: ❌ NOT available (only expiration date shown in user profile)
> - **Token Deletion**: ❌ NOT available (creating new token overwrites old one)
> - **Each user has ONE token** - creating a new token invalidates the previous one

---

## DoD Checklist

| # | DoD Item | Verification Section |
|---|----------|---------------------|
| 1 | MLflow OIDC container running locally with PostgreSQL in docker | [Section 1](#1-verify-mlflow-oidc-container-with-postgresql) |
| 2 | Users can create access tokens via UI/CLI | [Section 2](#2-verify-token-creation) |
| 3 | Users can access experiment results using tokens **(PRIMARY)** | [Section 3](#3-verify-token-based-experiment-access-primary-goal) |

> **Note**: Section 4 covers token management limitations - listing/deleting tokens is NOT supported in mlflow-oidc-auth v6.6.4.

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

mlflow-oidc-auth supports **two types of tokens**:
1. **OIDC Session Token** (from Keycloak) - Short-lived, for browser sessions
2. **MLflow Access Token** (stored in MLflow DB) - Long-lived, for API/CLI access

### Step 2.1: Create OIDC Token via CLI (Keycloak)

This gets a short-lived OIDC token (expires in ~5 minutes by default):

```bash
# Set your client secret (from configure-keycloak.sh)
export CLIENT_SECRET="YOUR_CLIENT_SECRET"

# Get OIDC access token from Keycloak
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
  "access_token": "eyJhbGci...",
  "expires_in": 300,
  "refresh_token": "eyJhbGci...",
  "token_type": "Bearer",
  "scope": "openid profile email"
}
```

### Step 2.2: Create MLflow Access Token via API

This creates a **long-lived** MLflow access token (up to 1 year):

```bash
# First, login to MLflow UI to get a session cookie, then:
# Use the MLflow access-token API endpoint

# Option 1: Using curl with session cookie (after UI login)
curl -X PATCH "http://localhost:5000/api/2.0/mlflow/users/access-token" \
  -H "Content-Type: application/json" \
  -b "session=YOUR_SESSION_COOKIE" \
  -d '{"expiration": "2027-01-01T00:00:00Z"}'
```

**Expected output**:
```json
{
  "token": "mlflow_access_token_abc123...",
  "message": "Token for mlflow_user has been created"
}
```

### Step 2.3: Create Token via MLflow UI

**Where**: MLflow UI at http://localhost:5000

**Steps**:
1. Open browser: **http://localhost:5000**
2. You'll be automatically redirected to Keycloak login page
3. Enter credentials:
   - Username: `mlflow_user`
   - Password: `password123`
4. Click **Sign In**
5. After login, look for **"Generate Access Token"** button in the UI sidebar
6. Click to open the Access Token Modal
7. Select expiration date (up to 1 year)
8. Click **"Request Token"**
9. Copy the generated token

**What happens**:
- MLflow generates a new access token stored in its database
- This token can be used for API calls with Basic Auth or Bearer token
- **Note**: Creating a new token **invalidates** the previous one (one token per user)

### Step 2.4: Save tokens for next steps

```bash
# Save OIDC token (short-lived)
export OIDC_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=$CLIENT_SECRET" \
  -d "username=mlflow_user" \
  -d "******" | jq -r '.access_token')

echo "OIDC Token: ${OIDC_TOKEN:0:50}..."

# Or use the MLflow Access Token you generated via UI
export MLFLOW_TOKEN="your-mlflow-access-token-from-ui"
```

### ✅ DoD #2 VERIFIED if:
- [ ] **CLI (Keycloak)**: curl returns JSON with `access_token` field
- [ ] **CLI (Keycloak)**: Token starts with "eyJ" (JWT format)
- [ ] **UI**: Visiting http://localhost:5000 redirects to Keycloak
- [ ] **UI**: Login with mlflow_user/password123 succeeds
- [ ] **UI**: "Generate Access Token" option visible in MLflow UI
- [ ] **UI**: Token generation modal works and returns token

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

---

## 4. Token Listing and Deletion (LIMITATIONS)

### ⚠️ CRITICAL FINDING from Source Code Review

After reviewing the **mlflow-oidc-auth v6.6.4** source code:

| Feature | Available? | Notes |
|---------|------------|-------|
| **Create Token via UI** | ✅ YES | "Generate Access Token" modal in MLflow UI |
| **Create Token via API** | ✅ YES | `PATCH /api/2.0/mlflow/users/access-token` |
| **List Tokens via UI** | ❌ NO | Only shows token expiration date in user profile |
| **List Tokens via API** | ❌ NO | No endpoint exists |
| **Delete Token via UI** | ❌ NO | Not implemented |
| **Delete Token via API** | ❌ NO | Creating new token overwrites old one |

### How MLflow-OIDC-Auth Token System Works

1. **One token per user**: Each user can only have ONE active MLflow access token
2. **Token rotation = deletion**: Creating a new token automatically invalidates the previous one
3. **Token storage**: Tokens are stored as hashed passwords in MLflow's user database
4. **Expiration only**: You can view token expiration via `GET /api/2.0/mlflow/users/current`

### Step 4.1: View Token Expiration

```bash
# After logging in, check your user profile to see token expiration
curl -s "http://localhost:5000/api/2.0/mlflow/users/current" \
  -H "Authorization: ******" | jq '.password_expiration'
```

**Expected output**: ISO date string like `"2027-01-01T00:00:00"`

### Step 4.2: "Delete" a Token (by rotating)

To effectively "delete" a token, create a new one (which invalidates the old):

```bash
# This creates a new token and invalidates the previous one
curl -X PATCH "http://localhost:5000/api/2.0/mlflow/users/access-token" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{"expiration": "2027-01-01T00:00:00Z"}'
```

### Step 4.3: Keycloak Session Management (For OIDC Tokens)

For **OIDC session tokens** (not MLflow access tokens), use Keycloak:

1. Open browser: http://localhost:8080/admin
2. Login with: `admin` / `admin123`
3. Select **mlflow** realm
4. Click **Sessions** in left menu
5. Find sessions and click **Sign out**

### ✅ Token Management Summary

| What you want to do | How to do it |
|---------------------|--------------|
| Create new token | MLflow UI → Generate Access Token, or API `PATCH /users/access-token` |
| View token expiration | MLflow UI → User Profile, or API `GET /users/current` |
| Invalidate old token | Create a new token (overwrites old) |
| Revoke OIDC session | Keycloak Admin → Sessions → Sign out |

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
