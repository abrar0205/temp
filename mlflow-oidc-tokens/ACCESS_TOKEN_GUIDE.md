# MLflow OIDC Access Token Guide

This guide documents how to use access tokens with the MLflow OIDC deployment for programmatic access to experiment results.

## Overview

The MLflow OIDC setup (located in `Mlflow_docker/`) uses Keycloak as the identity provider. Users can:
1. **Create access tokens** via Keycloak token endpoint or UI
2. **Access experiment results** using tokens with the MLflow Python SDK or REST API
3. **List and delete tokens** via Keycloak Admin Console/API

## Prerequisites

- MLflow OIDC deployment running (see `Mlflow_docker/readme.md`)
- Keycloak realm `mlflow` configured with client `mlflow`
- User account in Keycloak with `mlflow_users` or `mlflow_admins` group membership

## Creating Access Tokens

### Method 1: Password Grant (CLI)

Get an access token using your Keycloak credentials:

```bash
# Get access token
ACCESS_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=${OIDC_CLIENT_SECRET}" \
  -d "username=mlflow_user" \
  -d "******" | jq -r '.access_token')

echo "Access Token: $ACCESS_TOKEN"
```

### Method 2: Service Account Token (for automation)

For automated systems, use service account credentials:

```bash
# Get service account token (client credentials grant)
ACCESS_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=mlflow" \
  -d "client_secret=${OIDC_CLIENT_SECRET}" | jq -r '.access_token')
```

> **Note**: Service accounts require additional Keycloak configuration (enable "Service Accounts Enabled" in client settings).

### Method 3: Keycloak Admin Console

1. Go to http://localhost:8080/admin
2. Select the `mlflow` realm
3. Navigate to Clients → `mlflow` → Credentials
4. Generate a new client secret if needed

## Accessing Experiment Results Using Tokens

### Using MLflow Python SDK

```python
import os
import mlflow
import requests

def get_access_token(username, password, client_secret):
    """Get access token from Keycloak."""
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
    response.raise_for_status()
    return response.json()["access_token"]

# Get token
token = get_access_token("mlflow_user", "password123", os.getenv("OIDC_CLIENT_SECRET"))

# Configure MLflow to use the token
os.environ["MLFLOW_TRACKING_TOKEN"] = token
mlflow.set_tracking_uri("http://localhost:5000")

# Now you can access experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")

# Search runs
runs = mlflow.search_runs(experiment_ids=["0"])
print(runs)

# Log metrics (if you have write permissions)
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_param("learning_rate", 0.01)
```

### Using REST API with curl

```bash
# Set your access token
export ACCESS_TOKEN="your-token-here"

# List experiments
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'

# Get specific experiment
curl "http://localhost:5000/api/2.0/mlflow/experiments/get?experiment_id=0" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Search runs in an experiment
curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["0"]}'

# Create a new experiment
curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-new-experiment"}'
```

## Listing and Deleting Tokens

### ⚠️ Important Note on Token Management

**MLflow OIDC Auth does NOT provide a UI for listing/deleting tokens.** Token management is handled entirely through Keycloak.

As of mlflow-oidc-auth version 5.6.x:
- No built-in token listing UI
- No built-in token revocation UI
- Tokens are OAuth2 tokens managed by Keycloak

### Listing Active Sessions/Tokens via Keycloak Admin Console

1. Go to http://localhost:8080/admin
2. Select the `mlflow` realm
3. Navigate to **Sessions** in the left menu
4. View all active sessions across users

### Listing Tokens via Keycloak Admin API

```bash
# Get admin token first
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin" \
  -d "******" \
  -d "grant_type=password" \
  -d "client_id=admin-cli" | jq -r '.access_token')

# Get user ID
USER_ID=$(curl -s "http://localhost:8080/admin/realms/mlflow/users?username=mlflow_user" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq -r '.[0].id')

# List user's sessions (tokens)
curl -s "http://localhost:8080/admin/realms/mlflow/users/$USER_ID/sessions" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq

# List all client sessions
CLIENT_UUID=$(curl -s "http://localhost:8080/admin/realms/mlflow/clients?clientId=mlflow" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq -r '.[0].id')

curl -s "http://localhost:8080/admin/realms/mlflow/clients/$CLIENT_UUID/user-sessions" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq
```

### Deleting/Revoking Tokens

#### Via Keycloak Admin Console
1. Go to Sessions
2. Find the session to revoke
3. Click "Sign out" or "Logout all"

#### Via Keycloak Admin API

```bash
# Logout a specific user (revokes all their tokens)
curl -X POST "http://localhost:8080/admin/realms/mlflow/users/$USER_ID/logout" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Revoke a specific session
SESSION_ID="session-id-here"
curl -X DELETE "http://localhost:8080/admin/realms/mlflow/sessions/$SESSION_ID" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

#### Via Token Revocation Endpoint (Self-revocation)

```bash
# Revoke your own token
curl -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/revoke" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=mlflow" \
  -d "client_secret=${OIDC_CLIENT_SECRET}" \
  -d "token=$ACCESS_TOKEN"
```

## Token Management Capabilities Summary

| Feature | MLflow OIDC UI | Keycloak Admin Console | Keycloak Admin API |
|---------|----------------|------------------------|-------------------|
| Create Token | ❌ | ❌ (use token endpoint) | ✅ |
| Use Token | ✅ | N/A | N/A |
| List Tokens | ❌ | ✅ (Sessions view) | ✅ |
| Revoke Token | ❌ | ✅ | ✅ |
| Token Expiry Config | ❌ | ✅ | ✅ |

### Configuring Token Expiration

In Keycloak Admin Console:
1. Go to Realm Settings → Tokens
2. Configure:
   - **Access Token Lifespan** (default: 5 minutes)
   - **SSO Session Idle** (default: 30 minutes)
   - **SSO Session Max** (default: 10 hours)

For longer-lived tokens, consider using refresh tokens:

```bash
# Get tokens with refresh token
TOKENS=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=mlflow" \
  -d "client_secret=${OIDC_CLIENT_SECRET}" \
  -d "username=mlflow_user" \
  -d "******")

ACCESS_TOKEN=$(echo $TOKENS | jq -r '.access_token')
REFRESH_TOKEN=$(echo $TOKENS | jq -r '.refresh_token')

# Refresh the access token when it expires
NEW_TOKENS=$(curl -s -X POST "http://localhost:8080/realms/mlflow/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=refresh_token" \
  -d "client_id=mlflow" \
  -d "client_secret=${OIDC_CLIENT_SECRET}" \
  -d "refresh_token=$REFRESH_TOKEN")

NEW_ACCESS_TOKEN=$(echo $NEW_TOKENS | jq -r '.access_token')
```

## Future: Token Management in Newer mlflow-oidc-auth Versions

As of the current version (5.6.x), mlflow-oidc-auth does **not** provide:
- Token listing UI
- Token revocation UI
- Personal access tokens (like GitHub PATs)

Check the [mlflow-oidc-auth releases](https://github.com/mlflow-oidc/mlflow-oidc-auth/releases) for updates on these features.

For enterprise token management needs, consider:
1. Using Keycloak's built-in session management
2. Implementing a custom token proxy
3. Using Keycloak's offline tokens for long-lived access

## Troubleshooting

### "Invalid token" error
- Token may have expired (default: 5 minutes)
- Use refresh token to get a new access token
- Check token was issued for the correct realm/client

### "Unauthorized" error when using SDK
- Ensure `MLFLOW_TRACKING_TOKEN` environment variable is set
- Verify token hasn't expired
- Check user has correct group membership in Keycloak

### Token not being recognized
- Ensure you're using `Bearer` prefix in Authorization header
- Check `OIDC_CLIENT_ID` and `OIDC_CLIENT_SECRET` match between token request and MLflow config

## References

- [MLflow OIDC Auth Documentation](https://github.com/mlflow-oidc/mlflow-oidc-auth)
- [Keycloak Admin REST API](https://www.keycloak.org/docs-api/23.0.0/rest-api/)
- [OAuth 2.0 Token Endpoint](https://www.keycloak.org/docs/latest/securing_apps/index.html#token-endpoint)
- Existing setup: `Mlflow_docker/` directory on main branch
