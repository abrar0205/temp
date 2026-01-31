#!/bin/bash
# Keycloak Configuration Script
# Sets up MLflow realm, client, groups, and test user
#
# Usage: ./configure-keycloak.sh
#
# This script will:
# 1. Create 'mlflow' realm
# 2. Create 'mlflow-client' OIDC client
# 3. Create user groups (mlflow_admins, mlflow_users)
# 4. Create a test user (mlflow_user)
# 5. Output the client secret for .env configuration

set -e

# Configuration (can be overridden via environment variables)
KEYCLOAK_URL="${KEYCLOAK_EXTERNAL_URL:-http://localhost:8080}"
KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"
KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:-admin}"
REALM_NAME="${KEYCLOAK_REALM:-mlflow}"
CLIENT_ID="${OIDC_CLIENT_ID:-mlflow-client}"
MLFLOW_URL="${MLFLOW_EXTERNAL_URL:-http://localhost:5000}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Wait for Keycloak to be ready
print_status "Waiting for Keycloak to be ready..."
max_attempts=60
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -sf "${KEYCLOAK_URL}/health/ready" > /dev/null 2>&1; then
        print_status "Keycloak is ready!"
        break
    fi
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        print_error "Keycloak did not become ready in time"
        exit 1
    fi
    sleep 2
done

# Get admin access token
print_status "Obtaining admin access token..."
ADMIN_TOKEN=$(curl -sf -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${KEYCLOAK_ADMIN}" \
    -d "password=${KEYCLOAK_ADMIN_PASSWORD}" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" | jq -r '.access_token')

if [ -z "$ADMIN_TOKEN" ] || [ "$ADMIN_TOKEN" == "null" ]; then
    print_error "Failed to obtain admin token. Check Keycloak admin credentials."
    exit 1
fi

print_status "Admin token obtained successfully"

# Function to make authenticated API calls
keycloak_api() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -n "$data" ]; then
        curl -sf -X "$method" "${KEYCLOAK_URL}/admin/realms${endpoint}" \
            -H "Authorization: Bearer ${ADMIN_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$data"
    else
        curl -sf -X "$method" "${KEYCLOAK_URL}/admin/realms${endpoint}" \
            -H "Authorization: Bearer ${ADMIN_TOKEN}" \
            -H "Content-Type: application/json"
    fi
}

# Create realm
print_status "Creating realm '${REALM_NAME}'..."
REALM_EXISTS=$(keycloak_api GET "" 2>/dev/null | jq -r ".[] | select(.realm==\"${REALM_NAME}\") | .realm" || echo "")

if [ "$REALM_EXISTS" == "$REALM_NAME" ]; then
    print_warning "Realm '${REALM_NAME}' already exists, skipping creation"
else
    keycloak_api POST "" '{
        "realm": "'"${REALM_NAME}"'",
        "enabled": true,
        "registrationAllowed": false,
        "loginWithEmailAllowed": true,
        "duplicateEmailsAllowed": false,
        "resetPasswordAllowed": true,
        "editUsernameAllowed": false,
        "bruteForceProtected": true
    }' || print_warning "Failed to create realm (may already exist)"
    print_status "Realm '${REALM_NAME}' created"
fi

# Create client
print_status "Creating OIDC client '${CLIENT_ID}'..."
CLIENT_EXISTS=$(keycloak_api GET "/${REALM_NAME}/clients" 2>/dev/null | jq -r ".[] | select(.clientId==\"${CLIENT_ID}\") | .clientId" || echo "")

if [ "$CLIENT_EXISTS" == "$CLIENT_ID" ]; then
    print_warning "Client '${CLIENT_ID}' already exists"
    # Get existing client ID (UUID)
    CLIENT_UUID=$(keycloak_api GET "/${REALM_NAME}/clients" | jq -r ".[] | select(.clientId==\"${CLIENT_ID}\") | .id")
else
    keycloak_api POST "/${REALM_NAME}/clients" '{
        "clientId": "'"${CLIENT_ID}"'",
        "name": "MLflow OIDC Client",
        "description": "MLflow tracking server OIDC authentication",
        "enabled": true,
        "clientAuthenticatorType": "client-secret",
        "redirectUris": [
            "'"${MLFLOW_URL}"'/*",
            "http://localhost:5000/*"
        ],
        "webOrigins": [
            "'"${MLFLOW_URL}"'",
            "http://localhost:5000"
        ],
        "standardFlowEnabled": true,
        "directAccessGrantsEnabled": true,
        "serviceAccountsEnabled": false,
        "publicClient": false,
        "protocol": "openid-connect",
        "attributes": {
            "post.logout.redirect.uris": "'"${MLFLOW_URL}"'/*"
        }
    }'
    print_status "Client '${CLIENT_ID}' created"
    
    # Get the new client's UUID
    CLIENT_UUID=$(keycloak_api GET "/${REALM_NAME}/clients" | jq -r ".[] | select(.clientId==\"${CLIENT_ID}\") | .id")
fi

# Get client secret
print_status "Retrieving client secret..."
CLIENT_SECRET=$(keycloak_api GET "/${REALM_NAME}/clients/${CLIENT_UUID}/client-secret" | jq -r '.value')

if [ -z "$CLIENT_SECRET" ] || [ "$CLIENT_SECRET" == "null" ]; then
    # Generate new secret
    keycloak_api POST "/${REALM_NAME}/clients/${CLIENT_UUID}/client-secret" '{}'
    CLIENT_SECRET=$(keycloak_api GET "/${REALM_NAME}/clients/${CLIENT_UUID}/client-secret" | jq -r '.value')
fi

# Create groups
print_status "Creating user groups..."

for group_name in "mlflow_admins" "mlflow_users"; do
    GROUP_EXISTS=$(keycloak_api GET "/${REALM_NAME}/groups" 2>/dev/null | jq -r ".[] | select(.name==\"${group_name}\") | .name" || echo "")
    
    if [ "$GROUP_EXISTS" == "$group_name" ]; then
        print_warning "Group '${group_name}' already exists"
    else
        keycloak_api POST "/${REALM_NAME}/groups" '{"name": "'"${group_name}"'"}'
        print_status "Group '${group_name}' created"
    fi
done

# Get group IDs
ADMIN_GROUP_ID=$(keycloak_api GET "/${REALM_NAME}/groups" | jq -r '.[] | select(.name=="mlflow_admins") | .id')
USERS_GROUP_ID=$(keycloak_api GET "/${REALM_NAME}/groups" | jq -r '.[] | select(.name=="mlflow_users") | .id')

# Create test user
TEST_USERNAME="mlflow_user"
TEST_PASSWORD="password123"

print_status "Creating test user '${TEST_USERNAME}'..."
USER_EXISTS=$(keycloak_api GET "/${REALM_NAME}/users?username=${TEST_USERNAME}" 2>/dev/null | jq -r '.[0].username' || echo "")

if [ "$USER_EXISTS" == "$TEST_USERNAME" ]; then
    print_warning "User '${TEST_USERNAME}' already exists"
    USER_ID=$(keycloak_api GET "/${REALM_NAME}/users?username=${TEST_USERNAME}" | jq -r '.[0].id')
else
    keycloak_api POST "/${REALM_NAME}/users" '{
        "username": "'"${TEST_USERNAME}"'",
        "email": "'"${TEST_USERNAME}"'@example.com",
        "emailVerified": true,
        "enabled": true,
        "firstName": "MLflow",
        "lastName": "User",
        "credentials": [{
            "type": "password",
            "value": "'"${TEST_PASSWORD}"'",
            "temporary": false
        }]
    }'
    print_status "User '${TEST_USERNAME}' created"
    
    # Get user ID
    USER_ID=$(keycloak_api GET "/${REALM_NAME}/users?username=${TEST_USERNAME}" | jq -r '.[0].id')
fi

# Add user to groups
if [ -n "$USER_ID" ] && [ -n "$USERS_GROUP_ID" ]; then
    print_status "Adding user to 'mlflow_users' group..."
    keycloak_api PUT "/${REALM_NAME}/users/${USER_ID}/groups/${USERS_GROUP_ID}" '{}' 2>/dev/null || true
fi

# Output results
echo ""
echo "=========================================="
echo -e "${GREEN}Keycloak Configuration Complete!${NC}"
echo "=========================================="
echo ""
echo "Realm: ${REALM_NAME}"
echo "Client ID: ${CLIENT_ID}"
echo -e "Client Secret: ${YELLOW}${CLIENT_SECRET}${NC}"
echo ""
echo "Test User:"
echo "  Username: ${TEST_USERNAME}"
echo "  Password: ${TEST_PASSWORD}"
echo ""
echo "Groups:"
echo "  - mlflow_admins (admin access)"
echo "  - mlflow_users (regular access)"
echo ""
echo "=========================================="
echo -e "${YELLOW}IMPORTANT: Update your .env file with:${NC}"
echo ""
echo "OIDC_CLIENT_SECRET=${CLIENT_SECRET}"
echo ""
echo "Then restart MLflow:"
echo "  docker compose restart mlflow"
echo "=========================================="
