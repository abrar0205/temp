#!/bin/bash
# Keycloak Configuration Script for MLflow OIDC
# This script automates the setup of Keycloak realm, client, groups, and test user

set -e

# Configuration (can be overridden by environment variables)
KEYCLOAK_URL="${KEYCLOAK_URL:-http://localhost:8080}"
KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"
KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:-admin}"
REALM_NAME="${KEYCLOAK_REALM:-mlflow}"
CLIENT_ID="${OIDC_CLIENT_ID:-mlflow}"
MLFLOW_URL="${MLFLOW_URL:-http://localhost:5000}"
ADMIN_GROUP="${OIDC_ADMIN_GROUP_NAME:-mlflow_admins}"
USER_GROUP="${OIDC_GROUP_NAME:-mlflow_users}"

# Test user configuration
TEST_USERNAME="${TEST_USERNAME:-mlflow_user}"
TEST_PASSWORD="${TEST_PASSWORD:-password123}"
TEST_EMAIL="${TEST_EMAIL:-mlflow_user@example.com}"
TEST_FIRSTNAME="${TEST_FIRSTNAME:-MLflow}"
TEST_LASTNAME="${TEST_LASTNAME:-User}"

echo "=============================================="
echo "Keycloak Configuration for MLflow OIDC"
echo "=============================================="
echo "Keycloak URL: $KEYCLOAK_URL"
echo "Realm: $REALM_NAME"
echo "Client ID: $CLIENT_ID"
echo "MLflow URL: $MLFLOW_URL"
echo "=============================================="

# Wait for Keycloak to be ready
echo ""
echo "Waiting for Keycloak to be ready..."
max_attempts=30
attempt=0
while [ "$attempt" -lt "$max_attempts" ]; do
    if curl -s "$KEYCLOAK_URL/health/ready" | grep -q "UP"; then
        echo "Keycloak is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts: Keycloak not ready yet, waiting..."
    sleep 5
done

if [ "$attempt" -eq "$max_attempts" ]; then
    echo "ERROR: Keycloak did not become ready in time"
    exit 1
fi

# Get admin access token
echo ""
echo "Getting admin access token..."
ADMIN_TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$KEYCLOAK_ADMIN" \
    -d "password=$KEYCLOAK_ADMIN_PASSWORD" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" | jq -r '.access_token')

if [ "$ADMIN_TOKEN" == "null" ] || [ -z "$ADMIN_TOKEN" ]; then
    echo "ERROR: Failed to get admin token"
    exit 1
fi
echo "Admin token obtained successfully."

# Create realm
echo ""
echo "Creating realm '$REALM_NAME'..."
REALM_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    "$KEYCLOAK_URL/admin/realms/$REALM_NAME")

if [ "$REALM_EXISTS" == "404" ]; then
    curl -s -X POST "$KEYCLOAK_URL/admin/realms" \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"realm\": \"$REALM_NAME\",
            \"enabled\": true,
            \"displayName\": \"MLflow\",
            \"registrationAllowed\": false,
            \"loginWithEmailAllowed\": true,
            \"duplicateEmailsAllowed\": false,
            \"resetPasswordAllowed\": true,
            \"editUsernameAllowed\": false,
            \"bruteForceProtected\": true
        }"
    echo "Realm '$REALM_NAME' created."
else
    echo "Realm '$REALM_NAME' already exists."
fi

# Create client
echo ""
echo "Creating client '$CLIENT_ID'..."
CLIENT_EXISTS=$(curl -s \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients?clientId=$CLIENT_ID" | jq 'length')

if [ "$CLIENT_EXISTS" == "0" ]; then
    # Generate a random client secret (use hex encoding for consistent length)
    CLIENT_SECRET=$(openssl rand -hex 24)
    
    curl -s -X POST "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients" \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"clientId\": \"$CLIENT_ID\",
            \"name\": \"MLflow Tracking Server\",
            \"enabled\": true,
            \"protocol\": \"openid-connect\",
            \"publicClient\": false,
            \"clientAuthenticatorType\": \"client-secret\",
            \"secret\": \"$CLIENT_SECRET\",
            \"redirectUris\": [
                \"$MLFLOW_URL/*\",
                \"http://localhost:5000/*\"
            ],
            \"webOrigins\": [
                \"$MLFLOW_URL\",
                \"http://localhost:5000\"
            ],
            \"standardFlowEnabled\": true,
            \"directAccessGrantsEnabled\": true,
            \"serviceAccountsEnabled\": false,
            \"attributes\": {
                \"access.token.lifespan\": \"3600\"
            }
        }"
    echo "Client '$CLIENT_ID' created."
else
    # Get existing client secret
    CLIENT_UUID=$(curl -s \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients?clientId=$CLIENT_ID" | jq -r '.[0].id')
    
    CLIENT_SECRET=$(curl -s \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients/$CLIENT_UUID/client-secret" | jq -r '.value')
    
    echo "Client '$CLIENT_ID' already exists."
fi

# Create groups
echo ""
echo "Creating groups..."

create_group() {
    local group_name=$1
    local group_exists
    group_exists=$(curl -s \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        "$KEYCLOAK_URL/admin/realms/$REALM_NAME/groups?search=$group_name" | jq 'length')
    
    if [ "$group_exists" == "0" ]; then
        curl -s -X POST "$KEYCLOAK_URL/admin/realms/$REALM_NAME/groups" \
            -H "Authorization: Bearer $ADMIN_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"$group_name\"}"
        echo "  Group '$group_name' created."
    else
        echo "  Group '$group_name' already exists."
    fi
}

create_group "$ADMIN_GROUP"
create_group "$USER_GROUP"

# Create test user
echo ""
echo "Creating test user '$TEST_USERNAME'..."
USER_EXISTS=$(curl -s \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    "$KEYCLOAK_URL/admin/realms/$REALM_NAME/users?username=$TEST_USERNAME" | jq 'length')

if [ "$USER_EXISTS" == "0" ]; then
    # Create user
    curl -s -X POST "$KEYCLOAK_URL/admin/realms/$REALM_NAME/users" \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"username\": \"$TEST_USERNAME\",
            \"email\": \"$TEST_EMAIL\",
            \"firstName\": \"$TEST_FIRSTNAME\",
            \"lastName\": \"$TEST_LASTNAME\",
            \"enabled\": true,
            \"emailVerified\": true,
            \"credentials\": [{
                \"type\": \"password\",
                \"value\": \"$TEST_PASSWORD\",
                \"temporary\": false
            }]
        }"
    echo "Test user '$TEST_USERNAME' created."
    
    # Get user ID
    USER_ID=$(curl -s \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        "$KEYCLOAK_URL/admin/realms/$REALM_NAME/users?username=$TEST_USERNAME" | jq -r '.[0].id')
    
    # Get user group ID
    GROUP_ID=$(curl -s \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        "$KEYCLOAK_URL/admin/realms/$REALM_NAME/groups?search=$USER_GROUP" | jq -r '.[0].id')
    
    # Add user to group
    if [ "$GROUP_ID" != "null" ] && [ -n "$GROUP_ID" ]; then
        curl -s -X PUT "$KEYCLOAK_URL/admin/realms/$REALM_NAME/users/$USER_ID/groups/$GROUP_ID" \
            -H "Authorization: Bearer $ADMIN_TOKEN"
        echo "Test user added to '$USER_GROUP' group."
    fi
else
    echo "Test user '$TEST_USERNAME' already exists."
fi

# Print configuration summary
echo ""
echo "=============================================="
echo "Configuration Complete!"
echo "=============================================="
echo ""
echo "OIDC Client Secret:"
echo "  $CLIENT_SECRET"
echo ""
echo "Update your .env file with:"
echo "  OIDC_CLIENT_SECRET=$CLIENT_SECRET"
echo ""
echo "Then restart MLflow:"
echo "  docker compose restart mlflow"
echo ""
echo "Access URLs:"
echo "  MLflow UI: $MLFLOW_URL"
echo "  Keycloak Admin: $KEYCLOAK_URL"
echo ""
echo "Test User Credentials:"
echo "  Username: $TEST_USERNAME"
echo "  Password: $TEST_PASSWORD"
echo ""
echo "=============================================="
