#!/bin/bash
set -e

echo "Getting admin access token..."
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin" \
  -d "password=admin_password" \
  -d "grant_type=password" \
  -d "client_id=admin-cli" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

echo "Creating mlflow realm..."
curl -s -X POST "http://localhost:8080/admin/realms" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "realm": "mlflow",
    "enabled": true,
    "sslRequired": "none"
  }' || echo "Realm might already exist"

echo "Setting mlflow realm SSL requirement to NONE via database..."
docker exec mlflow-postgres-official psql -U keycloak -d keycloak -c "UPDATE realm SET ssl_required='NONE' WHERE name='mlflow';"

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

echo ""
echo "========================================="
echo "Keycloak Configuration Complete!"
echo "========================================="
echo "Client ID: mlflow"
echo "Client Secret: ${CLIENT_SECRET}"
echo "========================================="
echo ""
echo "Update your .env file with:"
echo "OIDC_CLIENT_SECRET=${CLIENT_SECRET}"
