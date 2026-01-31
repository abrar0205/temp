# MLflow OIDC Setup with Keycloak

This directory contains an alternative setup using Keycloak for full OIDC authentication with MLflow. This setup provides **proper token management** through Keycloak's admin console.

## 🔑 Key Advantages of OIDC Setup

1. **Token Listing** - View all active sessions/tokens in Keycloak admin console
2. **Token Deletion** - Revoke tokens/sessions from admin console
3. **Token Expiration** - Configure token lifetimes
4. **Single Sign-On** - Integrate with existing identity providers
5. **Audit Logging** - Track all authentication events

## 🚀 Quick Start

### 1. Start Services

```bash
docker-compose up -d
```

### 2. Configure Keycloak (First Time Setup)

1. Access Keycloak admin: http://localhost:8080
2. Login with `admin` / `admin`
3. Create a new realm named `mlflow`
4. Create a client:
   - Client ID: `mlflow`
   - Client Protocol: `openid-connect`
   - Root URL: `http://localhost:5000`
   - Valid Redirect URIs: `http://localhost:5000/*`
5. Create users and assign roles

### 3. Access MLflow

1. Navigate to http://localhost:5000
2. You will be redirected to Keycloak for login
3. After authentication, you'll be redirected back to MLflow

## 📋 Token Management in Keycloak

### Listing Active Tokens/Sessions

1. Go to Keycloak Admin Console: http://localhost:8080/admin
2. Select the `mlflow` realm
3. Navigate to **Sessions** in the left menu
4. View all active sessions across clients

Or use the Keycloak Admin REST API:

```bash
# Get admin access token
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin" \
  -d "password=admin" \
  -d "grant_type=password" \
  -d "client_id=admin-cli" | jq -r '.access_token')

# List all sessions for a user
curl -s "http://localhost:8080/admin/realms/mlflow/users/{user-id}/sessions" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Deleting/Revoking Tokens

**Via Admin Console:**
1. Go to **Sessions** → Select the session → Click **Logout**

**Via Admin API:**
```bash
# Logout all sessions for a user
curl -X POST "http://localhost:8080/admin/realms/mlflow/users/{user-id}/logout" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Logout specific session
curl -X DELETE "http://localhost:8080/admin/realms/mlflow/sessions/{session-id}" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Configuring Token Expiration

1. Go to Keycloak Admin Console
2. Navigate to **Realm Settings** → **Tokens**
3. Configure:
   - Access Token Lifespan (default: 5 minutes)
   - Refresh Token Lifespan
   - SSO Session Idle/Max

## 🔧 Using mlflow-oidc-auth Plugin

For full OIDC integration with MLflow, use the `mlflow-oidc-auth` plugin:

```bash
pip install mlflow-oidc-auth
```

Then start MLflow with:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow \
  --app-name oidc-auth
```

### Environment Variables

```bash
export OIDC_DISCOVERY_URL="http://localhost:8080/realms/mlflow/.well-known/openid-configuration"
export OIDC_CLIENT_ID="mlflow"
export OIDC_CLIENT_SECRET="your-client-secret"
export OIDC_REDIRECT_URI="http://localhost:5000/callback"
```

## 📊 Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Docker Network                                      │
│                                                                            │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐  │
│  │   PostgreSQL    │   │    Keycloak     │   │      MLflow Server      │  │
│  │   (MLflow DB)   │   │  (Port 8080)    │   │      (Port 5000)        │  │
│  │   Port 5432     │   │                 │   │                         │  │
│  └────────┬────────┘   │  ┌───────────┐  │   │  ┌─────────────────┐   │  │
│           │            │  │ Admin     │  │   │  │  OIDC Auth      │   │  │
│           │            │  │ Console   │  │◄──┼──│  Integration    │   │  │
│           │            │  └───────────┘  │   │  └─────────────────┘   │  │
│           │            │                 │   │                         │  │
│           │            │  ┌───────────┐  │   │  ┌─────────────────┐   │  │
│           └────────────┼──│ mlflow    │  │   │  │  Experiments    │   │  │
│                        │  │ realm     │  │   │  │  & Runs         │   │  │
│                        │  └───────────┘  │   │  └─────────────────┘   │  │
│                        └─────────────────┘   └─────────────────────────┘  │
│                                                                            │
│  ┌─────────────────┐                                                       │
│  │   PostgreSQL    │                                                       │
│  │  (Keycloak DB)  │                                                       │
│  └─────────────────┘                                                       │
└────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Token Flow

```
1. User accesses MLflow UI
         │
         ▼
2. MLflow redirects to Keycloak login
         │
         ▼
3. User authenticates with Keycloak
         │
         ▼
4. Keycloak issues tokens (access + refresh)
         │
         ▼
5. User is redirected back to MLflow with tokens
         │
         ▼
6. MLflow validates tokens with Keycloak
         │
         ▼
7. User can access experiments and log results
```

## 📝 Notes

- Keycloak provides enterprise-grade token management
- All sessions can be monitored and revoked centrally
- Supports integration with LDAP, SAML, and social login providers
- Audit logs available for compliance requirements
