# MLflow OIDC Access Token Guide

This guide explains **where** and **how** to manage access tokens for MLflow OIDC authentication.

> **Note**: The MLflow deployment exists at `Mlflow_docker/` on the main branch.

---

## Quick Reference

| Task | Where | How |
|------|-------|-----|
| Create Token | Keycloak Token Endpoint | `curl` command or login via UI |
| Use Token | MLflow API/SDK | `Authorization: Bearer <token>` header |
| List Tokens | Keycloak Admin Console | Sessions menu |
| Delete Tokens | Keycloak Admin Console | Sessions → Sign out |
| Configure Token Expiry | Keycloak Admin Console | Realm Settings → Tokens |

---

## 1. Creating Access Tokens

### Where
**Keycloak Token Endpoint**: `http://localhost:8080/realms/mlflow/protocol/openid-connect/token`

### How

**Option A: Using curl (CLI)**
```bash
curl -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "username=mlflow_user" \
  -d "password=password123"
```

Response contains:
```json
{
  "access_token": "eyJhbGciOiJS...",
  "refresh_token": "eyJhbGciOiJS...",
  "expires_in": 300
}
```

**Option B: Login via MLflow UI**
1. Go to http://localhost:5000
2. Click "Login with Keycloak"
3. Enter credentials
4. Token is automatically managed by the browser session

---

## 2. Using Tokens to Access Experiments

### Where
**MLflow API**: `http://localhost:5000/api/2.0/mlflow/`

### How

**List Experiments**
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Get Runs from Experiment**
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["0"]}'
```

**Create Experiment**
```bash
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-experiment"}'
```

---

## 3. Listing Tokens (Active Sessions)

### Where
**Keycloak Admin Console**: http://localhost:8080/admin

### How

1. Go to http://localhost:8080/admin
2. Login with admin credentials (from `.env` file)
3. Select **mlflow** realm (dropdown in top-left)
4. Click **Sessions** in the left menu
5. View all active sessions/tokens

![Sessions Location](https://www.keycloak.org/docs/latest/server_admin/images/sessions.png)

**Alternative: Using curl**
```bash
# First, get admin token
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=admin-cli" \
  -d "username=admin" \
  -d "password=YOUR_ADMIN_PASSWORD" | jq -r '.access_token')

# List all sessions for mlflow client
curl -s "http://localhost:8080/admin/realms/mlflow/clients" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq '.[].id'
```

---

## 4. Deleting/Revoking Tokens

### Where
**Keycloak Admin Console**: http://localhost:8080/admin → Sessions

### How

**Option A: Via Admin Console (Recommended)**
1. Go to http://localhost:8080/admin
2. Select **mlflow** realm
3. Click **Sessions**
4. Find the session to revoke
5. Click **Sign out** (for single session) or **Sign out all sessions**

**Option B: Revoke All Sessions for a User**
1. Go to **Users** in left menu
2. Click on the user
3. Go to **Sessions** tab
4. Click **Sign out all sessions**

**Option C: Using curl**
```bash
# Logout specific user (revokes all their tokens)
USER_ID="user-uuid-here"
curl -X POST "http://localhost:8080/admin/realms/mlflow/users/$USER_ID/logout" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## 5. Token Expiry Configuration

### Where
**Keycloak Admin Console**: http://localhost:8080/admin → Realm Settings → Tokens

### How

1. Go to http://localhost:8080/admin
2. Select **mlflow** realm
3. Click **Realm Settings** in left menu
4. Click **Tokens** tab
5. Configure:
   - **Access Token Lifespan**: How long access tokens are valid (default: 5 minutes)
   - **SSO Session Idle**: Idle timeout (default: 30 minutes)
   - **SSO Session Max**: Maximum session duration (default: 10 hours)

---

## Important Notes

### Token Listing/Deletion in MLflow OIDC

⚠️ **As of mlflow-oidc-auth version 5.6.x:**
- There is **NO built-in UI** in MLflow for listing tokens
- There is **NO built-in UI** in MLflow for deleting tokens
- All token management must be done through **Keycloak Admin Console**

Check [mlflow-oidc-auth releases](https://github.com/mlflow-oidc/mlflow-oidc-auth/releases) for updates on these features.

---

## Summary Table

| Operation | Where to Go | What to Click/Do |
|-----------|-------------|------------------|
| **Create Token** | Terminal | Run `curl` command to token endpoint |
| **Use Token** | Terminal/Code | Add `Authorization: Bearer <token>` header |
| **List Tokens** | http://localhost:8080/admin | Realm → Sessions |
| **Delete Token** | http://localhost:8080/admin | Sessions → Sign out |
| **Delete All User Tokens** | http://localhost:8080/admin | Users → [user] → Sessions → Sign out all |
| **Set Token Expiry** | http://localhost:8080/admin | Realm Settings → Tokens |

---

## URLs Reference

| Service | URL | Purpose |
|---------|-----|---------|
| MLflow UI | http://localhost:5000 | Experiment tracking UI |
| MLflow API | http://localhost:5000/api/2.0/mlflow/ | REST API |
| Keycloak Admin | http://localhost:8080/admin | Token & user management |
| Keycloak Token Endpoint | http://localhost:8080/realms/mlflow/protocol/openid-connect/token | Get access tokens |
| MinIO Console | http://localhost:9001 | Artifact storage |
