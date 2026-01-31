#!/bin/bash
set -e

KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:-admin123}"

echo "Waiting for Keycloak to be ready..."
sleep 10

echo "Getting admin access token..."
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin" \
  -d "password=${KEYCLOAK_ADMIN_PASSWORD}" \
  -d "grant_type=password" \
  -d "client_id=admin-cli" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$ADMIN_TOKEN" ]; then
    echo "Failed to get admin token. Check Keycloak admin credentials."
    exit 1
fi

echo "Creating mlflow realm..."
curl -s -X POST "http://localhost:8080/admin/realms" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "realm": "mlflow",
    "enabled": true,
    "sslRequired": "none"
  }' || echo "Realm might already exist"

echo "Creating mlflow-admins group..."
curl -s -X POST "http://localhost:8080/admin/realms/mlflow/groups" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name": "mlflow-admins"}' || echo "Group might already exist"

echo "Creating mlflow-users group..."
curl -s -X POST "http://localhost:8080/admin/realms/mlflow/groups" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name": "mlflow-users"}' || echo "Group might already exist"

echo "Creating mlflow client..."
curl -s -X POST "http://localhost:8080/admin/realms/mlflow/clients" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "clientId": "mlflow",
    "enabled": true,
    "clientAuthenticatorType": "client-secret",
    "redirectUris": ["http://localhost:5000/*"],
    "webOrigins": ["+"],
    "standardFlowEnabled": true,
    "directAccessGrantsEnabled": true,
    "serviceAccountsEnabled": true,
    "publicClient": false
  }' || echo "Client might already exist"

echo "Getting client secret..."
CLIENT_UUID=$(curl -s "http://localhost:8080/admin/realms/mlflow/clients?clientId=mlflow" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" | python3 -c "import sys, json; print(json.load(sys.stdin)[0]['id'])")

CLIENT_SECRET=$(curl -s "http://localhost:8080/admin/realms/mlflow/clients/${CLIENT_UUID}/client-secret" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" | python3 -c "import sys, json; print(json.load(sys.stdin)['value'])")

echo "Creating test user mlflow_user..."
curl -s -X POST "http://localhost:8080/admin/realms/mlflow/users" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "mlflow_user",
    "enabled": true,
    "emailVerified": true,
    "email": "mlflow_user@example.com",
    "firstName": "MLflow",
    "lastName": "User",
    "credentials": [{"type": "password", "value": "password123", "temporary": false}]
  }' || echo "User might already exist"

# Get user ID and add to groups
USER_ID=$(curl -s "http://localhost:8080/admin/realms/mlflow/users?username=mlflow_user" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data[0]['id'] if data else '')")

if [ -n "$USER_ID" ]; then
    # Get group IDs
    ADMIN_GROUP_ID=$(curl -s "http://localhost:8080/admin/realms/mlflow/groups?search=mlflow-admins" \
      -H "Authorization: Bearer ${ADMIN_TOKEN}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data[0]['id'] if data else '')")
    
    USERS_GROUP_ID=$(curl -s "http://localhost:8080/admin/realms/mlflow/groups?search=mlflow-users" \
      -H "Authorization: Bearer ${ADMIN_TOKEN}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data[0]['id'] if data else '')")
    
    # Add user to groups
    if [ -n "$ADMIN_GROUP_ID" ]; then
        echo "Adding user to mlflow-admins group..."
        curl -s -X PUT "http://localhost:8080/admin/realms/mlflow/users/${USER_ID}/groups/${ADMIN_GROUP_ID}" \
          -H "Authorization: Bearer ${ADMIN_TOKEN}"
    fi
    
    if [ -n "$USERS_GROUP_ID" ]; then
        echo "Adding user to mlflow-users group..."
        curl -s -X PUT "http://localhost:8080/admin/realms/mlflow/users/${USER_ID}/groups/${USERS_GROUP_ID}" \
          -H "Authorization: Bearer ${ADMIN_TOKEN}"
    fi
fi

echo ""
echo "========================================="
echo "Keycloak Configuration Complete!"
echo "========================================="
echo "Client ID: mlflow"
echo "Client Secret: ${CLIENT_SECRET}"
echo ""
echo "Test User:"
echo "  Username: mlflow_user"
echo "  Password: password123"
echo "========================================="
echo ""
echo "Update your .env file with:"
echo "OIDC_CLIENT_SECRET=${CLIENT_SECRET}"
echo ""
echo "Then restart MLflow with:"
echo "docker compose restart mlflow"
