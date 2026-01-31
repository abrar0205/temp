# MLflow OIDC Access Token Management Guide

This guide provides comprehensive documentation for managing access tokens in MLflow with OIDC (OpenID Connect) authentication, including login procedures, token listing, deletion operations, and container cleanup.

---

## Table of Contents

1. [OIDC Login Procedure](#oidc-login-procedure)
2. [Token Listing Methods](#token-listing-methods)
3. [Token Deletion Methods](#token-deletion-methods)
4. [MLflow OIDC UI Support for Token Management](#mlflow-oidc-ui-support-for-token-management)
5. [OIDC Container Cleanup](#oidc-container-cleanup)
6. [Troubleshooting](#troubleshooting)

---

## OIDC Login Procedure

### Prerequisites

Before logging in to MLflow with OIDC authentication, ensure you have:

- MLflow server configured with OIDC provider (e.g., Keycloak, Okta, Azure AD)
- Valid user credentials registered with the OIDC provider
- Network access to both MLflow server and OIDC provider endpoints

### Web UI Login

1. **Navigate to MLflow UI**
   
   Open your web browser and navigate to your MLflow tracking server URL:
   ```
   https://your-mlflow-server.example.com
   ```

2. **Initiate OIDC Login**
   
   You will be automatically redirected to the OIDC provider's login page, or click the "Login with SSO" or "Sign in with OIDC" button.

   ![OIDC Login Button](screenshots/oidc-login-button-placeholder.png)
   *Screenshot: MLflow login page with OIDC sign-in option*

3. **Enter Credentials**
   
   On the OIDC provider's login page:
   - Enter your username/email
   - Enter your password
   - Complete any MFA (Multi-Factor Authentication) if configured
   
   ![OIDC Provider Login](screenshots/oidc-provider-login-placeholder.png)
   *Screenshot: OIDC provider login form*

4. **Grant Authorization**
   
   If prompted, review and approve the requested permissions/scopes for MLflow access.

5. **Successful Login**
   
   After successful authentication, you will be redirected back to MLflow UI with an active session.
   
   ![MLflow Dashboard After Login](screenshots/mlflow-dashboard-placeholder.png)
   *Screenshot: MLflow dashboard after successful OIDC login*

### CLI Login

To authenticate via the MLflow CLI:

```bash
# Set the MLflow tracking URI
export MLFLOW_TRACKING_URI=https://your-mlflow-server.example.com

# Login using OIDC (browser-based flow)
mlflow login

# Alternatively, use environment variables for token-based auth
export MLFLOW_TRACKING_TOKEN=<your-access-token>
```

### Programmatic Login (Python)

```python
import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri("https://your-mlflow-server.example.com")

# If using token-based authentication
os.environ["MLFLOW_TRACKING_TOKEN"] = "<your-access-token>"

# Verify connection
client = mlflow.MlflowClient()
experiments = client.search_experiments()
print(f"Successfully connected. Found {len(experiments)} experiments.")
```

---

## Token Listing Methods

### Method 1: CLI-Based Token Listing

#### Using MLflow CLI (if supported by your MLflow version)

```bash
# List all personal access tokens
mlflow tokens list

# List tokens with detailed information
mlflow tokens list --verbose

# List tokens in JSON format for programmatic use
mlflow tokens list --output json
```

#### Using curl with REST API

```bash
# List all tokens for the current user
curl -X GET \
  "https://your-mlflow-server.example.com/api/2.0/mlflow/tokens" \
  -H "Authorization: Bearer <your-access-token>" \
  -H "Content-Type: application/json"
```

Example response:
```json
{
  "tokens": [
    {
      "token_id": "tok_abc123",
      "name": "ci-cd-pipeline",
      "created_at": "2024-01-15T10:30:00Z",
      "expires_at": "2024-07-15T10:30:00Z",
      "last_used_at": "2024-01-20T14:22:00Z",
      "scopes": ["experiments:read", "models:write"]
    },
    {
      "token_id": "tok_def456",
      "name": "local-development",
      "created_at": "2024-01-10T08:00:00Z",
      "expires_at": null,
      "last_used_at": "2024-01-19T09:15:00Z",
      "scopes": ["all"]
    }
  ]
}
```

### Method 2: API-Based Token Listing

#### Python SDK

```python
import requests
import os

# Configuration
MLFLOW_SERVER = "https://your-mlflow-server.example.com"
ACCESS_TOKEN = os.environ.get("MLFLOW_TRACKING_TOKEN")

def list_tokens():
    """List all access tokens for the current user."""
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(
        f"{MLFLOW_SERVER}/api/2.0/mlflow/tokens",
        headers=headers
    )
    
    if response.status_code == 200:
        tokens = response.json().get("tokens", [])
        print(f"Found {len(tokens)} token(s):")
        for token in tokens:
            print(f"  - ID: {token['token_id']}")
            print(f"    Name: {token['name']}")
            print(f"    Created: {token['created_at']}")
            print(f"    Expires: {token.get('expires_at', 'Never')}")
            print(f"    Last Used: {token.get('last_used_at', 'Never')}")
            print()
        return tokens
    else:
        print(f"Error listing tokens: {response.status_code}")
        print(response.text)
        return None

# Execute
tokens = list_tokens()
```

### Method 3: UI-Based Token Listing

If your MLflow installation supports token management via the UI:

1. **Navigate to User Settings**
   
   Click on your username/profile icon in the top-right corner of the MLflow UI.

2. **Open Access Tokens Section**
   
   Select "Access Tokens" or "API Tokens" from the dropdown menu.
   
   ![Access Tokens Menu](screenshots/access-tokens-menu-placeholder.png)
   *Screenshot: Navigating to Access Tokens in MLflow UI*

3. **View Token List**
   
   The token management page displays all your active tokens with:
   - Token name/description
   - Creation date
   - Expiration date (if applicable)
   - Last usage timestamp
   - Token scopes/permissions
   
   ![Token List View](screenshots/token-list-view-placeholder.png)
   *Screenshot: Access tokens list in MLflow UI*

---

## Token Deletion Methods

### Method 1: CLI-Based Token Deletion

#### Using MLflow CLI

```bash
# Delete a specific token by ID
mlflow tokens delete --token-id tok_abc123

# Delete a token by name
mlflow tokens delete --name "ci-cd-pipeline"

# Delete with confirmation prompt
mlflow tokens delete --token-id tok_abc123 --confirm

# Force delete without confirmation (use with caution)
mlflow tokens delete --token-id tok_abc123 --force
```

#### Using curl with REST API

```bash
# Delete a specific token
curl -X DELETE \
  "https://your-mlflow-server.example.com/api/2.0/mlflow/tokens/tok_abc123" \
  -H "Authorization: Bearer <your-access-token>" \
  -H "Content-Type: application/json"

# Delete with request body (alternative approach)
curl -X DELETE \
  "https://your-mlflow-server.example.com/api/2.0/mlflow/tokens" \
  -H "Authorization: Bearer <your-access-token>" \
  -H "Content-Type: application/json" \
  -d '{"token_id": "tok_abc123"}'
```

Expected response on successful deletion:
```json
{
  "status": "success",
  "message": "Token tok_abc123 has been deleted"
}
```

### Method 2: API-Based Token Deletion

#### Python SDK

```python
import requests
import os

# Configuration
MLFLOW_SERVER = "https://your-mlflow-server.example.com"
ACCESS_TOKEN = os.environ.get("MLFLOW_TRACKING_TOKEN")

def delete_token(token_id):
    """Delete an access token by ID."""
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.delete(
        f"{MLFLOW_SERVER}/api/2.0/mlflow/tokens/{token_id}",
        headers=headers
    )
    
    if response.status_code == 200:
        print(f"Successfully deleted token: {token_id}")
        return True
    elif response.status_code == 404:
        print(f"Token not found: {token_id}")
        return False
    else:
        print(f"Error deleting token: {response.status_code}")
        print(response.text)
        return False

def delete_all_tokens():
    """Delete all access tokens (use with caution)."""
    tokens = list_tokens()  # Using function from previous section
    
    if tokens:
        for token in tokens:
            delete_token(token['token_id'])
        print("All tokens deleted.")
    else:
        print("No tokens to delete.")

# Delete a specific token
delete_token("tok_abc123")

# Delete all tokens (use with caution)
# delete_all_tokens()
```

### Method 3: UI-Based Token Deletion

1. **Navigate to Access Tokens**
   
   Go to User Settings → Access Tokens (as described in the Token Listing section).

2. **Locate the Token to Delete**
   
   Find the token you want to delete in the list.

3. **Click Delete/Revoke Button**
   
   Click the delete icon (🗑️) or "Revoke" button next to the token.
   
   ![Token Delete Button](screenshots/token-delete-button-placeholder.png)
   *Screenshot: Delete button for a token in MLflow UI*

4. **Confirm Deletion**
   
   Confirm the deletion in the popup dialog.
   
   ![Token Delete Confirmation](screenshots/token-delete-confirm-placeholder.png)
   *Screenshot: Confirmation dialog for token deletion*

5. **Verify Deletion**
   
   The token should be removed from the list immediately.

---

## MLflow OIDC UI Support for Token Management

### Current State of MLflow OIDC Token Management UI

As of MLflow version 2.x and the MLflow OIDC authentication plugin, UI support for token management varies depending on the deployment:

#### MLflow OSS (Open Source)

- **Basic Authentication**: MLflow OSS supports basic token creation through the UI since version 2.5+
- **Token Listing**: Available in newer versions (2.8+)
- **Token Deletion**: Basic support available

#### MLflow with OIDC Plugin (mlflow-oidc-client)

The `mlflow-oidc-client` plugin provides:

| Feature | CLI Support | API Support | UI Support |
|---------|-------------|-------------|------------|
| Login/Logout | ✅ Yes | ✅ Yes | ✅ Yes |
| Token Creation | ✅ Yes | ✅ Yes | Partial |
| Token Listing | ✅ Yes | ✅ Yes | ⚠️ Limited |
| Token Deletion | ✅ Yes | ✅ Yes | ⚠️ Limited |
| Token Refresh | ✅ Yes | ✅ Yes | ✅ Auto |

#### Databricks-Managed MLflow

Databricks offers full UI support for:
- Personal Access Token (PAT) management
- Service Principal token management
- Token rotation and expiration policies

### Checking Your MLflow Version

```bash
# Check MLflow version
mlflow --version

# Check installed plugins
pip list | grep mlflow

# Check OIDC plugin version (if installed)
pip show mlflow-oidc-client
```

### Enabling UI Token Management (if available)

For MLflow deployments that support UI-based token management:

1. **Ensure Latest Version**
   ```bash
   pip install --upgrade mlflow mlflow-oidc-client
   ```

2. **Configure Server**
   
   Add to your MLflow server configuration:
   ```yaml
   # mlflow-config.yaml
   authentication:
     oidc:
       enabled: true
       provider_url: https://your-oidc-provider.com
       client_id: mlflow-client
       token_management_ui: true  # Enable UI if supported
   ```

3. **Restart MLflow Server**
   ```bash
   mlflow server --config mlflow-config.yaml
   ```

### Research: Newer MLflow OIDC Versions

Based on the latest MLflow releases and community contributions:

1. **MLflow 2.10+**: Improved authentication UI with better token management
2. **MLflow 2.12+**: Enhanced OIDC support with token introspection
3. **MLflow OIDC Plugin 0.2.0+**: May include expanded UI features

**Recommendation**: Check the official MLflow changelog and GitHub releases for the most current feature availability:
- [MLflow Releases](https://github.com/mlflow/mlflow/releases)
- [MLflow OIDC Plugin](https://github.com/mlflow/mlflow/tree/master/mlflow/server/auth)

---

## OIDC Container Cleanup

### Docker-Based MLflow OIDC Deployment Cleanup

#### Listing Running Containers

```bash
# List all MLflow-related containers
docker ps -a | grep -E "(mlflow|keycloak|oidc)"

# List with detailed formatting
docker ps -a --filter "name=mlflow" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"
```

#### Stopping Containers

```bash
# Stop MLflow server container
docker stop mlflow-server

# Stop OIDC provider container (e.g., Keycloak)
docker stop keycloak

# Stop all MLflow-related containers
docker stop $(docker ps -q --filter "name=mlflow")

# Stop containers using docker-compose
docker-compose -f docker-compose-mlflow.yml stop
```

#### Removing Containers

```bash
# Remove stopped MLflow container
docker rm mlflow-server

# Remove OIDC provider container
docker rm keycloak

# Force remove running container
docker rm -f mlflow-server

# Remove all MLflow-related containers
docker rm $(docker ps -aq --filter "name=mlflow")

# Remove containers using docker-compose
docker-compose -f docker-compose-mlflow.yml down
```

#### Complete Cleanup (Containers, Volumes, Networks)

```bash
# Full cleanup with docker-compose (recommended)
docker-compose -f docker-compose-mlflow.yml down -v --rmi local

# Manual full cleanup
# 1. Stop and remove containers
docker stop mlflow-server keycloak postgres-mlflow
docker rm mlflow-server keycloak postgres-mlflow

# 2. Remove associated volumes
docker volume rm mlflow_data keycloak_data postgres_data

# 3. Remove network
docker network rm mlflow-network

# 4. Remove images (optional)
docker rmi mlflow-server:latest keycloak:latest postgres:latest
```

### Kubernetes-Based MLflow OIDC Deployment Cleanup

```bash
# Delete MLflow deployment
kubectl delete deployment mlflow-server -n mlflow

# Delete OIDC/Keycloak deployment
kubectl delete deployment keycloak -n mlflow

# Delete services
kubectl delete service mlflow-server keycloak -n mlflow

# Delete secrets (tokens, credentials)
kubectl delete secret mlflow-oidc-secret -n mlflow

# Delete ConfigMaps
kubectl delete configmap mlflow-config -n mlflow

# Delete PersistentVolumeClaims
kubectl delete pvc mlflow-pvc keycloak-pvc -n mlflow

# Full namespace cleanup (CAUTION: deletes everything in namespace)
kubectl delete namespace mlflow
```

### Cleanup Scripts

#### Bash Script for Docker Cleanup

```bash
#!/bin/bash
# cleanup-mlflow-oidc.sh
# Complete cleanup script for MLflow OIDC Docker deployment

set -e

echo "=== MLflow OIDC Cleanup Script ==="

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose-mlflow.yml}"
CONTAINERS="mlflow-server keycloak postgres-mlflow"
VOLUMES="mlflow_data keycloak_data postgres_data"
NETWORK="mlflow-network"

# Function to confirm action
confirm() {
    read -p "$1 (y/N): " response
    case "$response" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) return 1 ;;
    esac
}

# Stop containers
echo "Stopping containers..."
if [ -f "$COMPOSE_FILE" ]; then
    docker-compose -f "$COMPOSE_FILE" stop 2>/dev/null || true
else
    for container in $CONTAINERS; do
        docker stop "$container" 2>/dev/null || true
    done
fi

# Remove containers
if confirm "Remove containers?"; then
    echo "Removing containers..."
    if [ -f "$COMPOSE_FILE" ]; then
        docker-compose -f "$COMPOSE_FILE" rm -f 2>/dev/null || true
    else
        for container in $CONTAINERS; do
            docker rm "$container" 2>/dev/null || true
        done
    fi
fi

# Remove volumes
if confirm "Remove volumes (WARNING: This will delete all data)?"; then
    echo "Removing volumes..."
    for volume in $VOLUMES; do
        docker volume rm "$volume" 2>/dev/null || true
    done
fi

# Remove network
if confirm "Remove network?"; then
    echo "Removing network..."
    docker network rm "$NETWORK" 2>/dev/null || true
fi

# Prune unused resources
if confirm "Prune unused Docker resources?"; then
    echo "Pruning unused resources..."
    docker system prune -f
fi

echo "=== Cleanup Complete ==="
```

#### Python Script for Cleanup

```python
#!/usr/bin/env python3
"""
MLflow OIDC Container Cleanup Script
Safely removes MLflow and OIDC-related Docker resources
"""

import subprocess
import sys
from typing import List, Optional

def run_command(cmd: List[str], check: bool = False) -> Optional[str]:
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Warning: Command failed: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        return None

def get_mlflow_containers() -> List[str]:
    """Get list of MLflow-related containers."""
    output = run_command([
        "docker", "ps", "-a", "--filter", "name=mlflow",
        "--format", "{{.Names}}"
    ])
    if output:
        return output.split('\n')
    return []

def stop_containers(containers: List[str]) -> None:
    """Stop specified containers."""
    for container in containers:
        print(f"Stopping container: {container}")
        run_command(["docker", "stop", container])

def remove_containers(containers: List[str]) -> None:
    """Remove specified containers."""
    for container in containers:
        print(f"Removing container: {container}")
        run_command(["docker", "rm", container])

def remove_volumes(volumes: List[str]) -> None:
    """Remove specified volumes."""
    for volume in volumes:
        print(f"Removing volume: {volume}")
        run_command(["docker", "volume", "rm", volume])

def cleanup(dry_run: bool = False) -> None:
    """Perform cleanup of MLflow OIDC resources."""
    containers = get_mlflow_containers()
    volumes = ["mlflow_data", "keycloak_data", "postgres_data"]
    
    print(f"Found {len(containers)} MLflow-related containers:")
    for c in containers:
        print(f"  - {c}")
    
    if dry_run:
        print("\nDry run - no changes made.")
        return
    
    if containers:
        stop_containers(containers)
        remove_containers(containers)
    
    # Uncomment to also remove volumes:
    # remove_volumes(volumes)
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    cleanup(dry_run=dry_run)
```

### Verifying Cleanup

After cleanup, verify all resources are removed:

```bash
# Verify no MLflow containers remain
docker ps -a | grep -E "(mlflow|keycloak)" && echo "Containers still exist!" || echo "All containers removed ✓"

# Verify volumes removed
docker volume ls | grep -E "(mlflow|keycloak)" && echo "Volumes still exist!" || echo "All volumes removed ✓"

# Verify network removed
docker network ls | grep mlflow && echo "Network still exists!" || echo "Network removed ✓"

# Check disk space recovered
df -h
```

---

## Troubleshooting

### Common Issues

#### 1. Cannot List Tokens - 401 Unauthorized

**Problem**: API calls return 401 error when listing tokens.

**Solution**:
```bash
# Refresh your access token
mlflow login --refresh

# Or set a new token
export MLFLOW_TRACKING_TOKEN=<new-token>
```

#### 2. Token Deletion Fails - 403 Forbidden

**Problem**: Unable to delete a token due to permission issues.

**Solution**:
- Verify you have permission to delete the token
- Admin tokens may require elevated privileges
- Check token ownership matches your user

#### 3. OIDC Provider Unreachable

**Problem**: Cannot authenticate because OIDC provider is down.

**Solution**:
```bash
# Check OIDC provider status
curl -I https://your-oidc-provider.com/.well-known/openid-configuration

# Check container status
docker ps | grep keycloak

# Restart if needed
docker restart keycloak
```

#### 4. Container Cleanup Fails - Volume in Use

**Problem**: Cannot remove volume because it's in use.

**Solution**:
```bash
# Find containers using the volume
docker ps -a --filter "volume=mlflow_data"

# Stop and remove those containers first
docker stop <container-id>
docker rm <container-id>

# Then remove the volume
docker volume rm mlflow_data
```

### Getting Help

- **MLflow Documentation**: [mlflow.org/docs](https://mlflow.org/docs/latest/index.html)
- **MLflow GitHub Issues**: [github.com/mlflow/mlflow/issues](https://github.com/mlflow/mlflow/issues)
- **OIDC Specification**: [openid.net/specs](https://openid.net/specs/openid-connect-core-1_0.html)

---

## Summary

This guide covered:

| Topic | CLI | API | UI |
|-------|-----|-----|-----|
| OIDC Login | ✅ `mlflow login` | ✅ Token-based | ✅ Browser redirect |
| Token Listing | ✅ `mlflow tokens list` | ✅ GET `/tokens` | ⚠️ Version dependent |
| Token Deletion | ✅ `mlflow tokens delete` | ✅ DELETE `/tokens/{id}` | ⚠️ Version dependent |
| Container Cleanup | ✅ `docker-compose down` | N/A | N/A |

**Key Takeaways**:
1. Always use CLI or API for reliable token management
2. Check your MLflow version for UI feature availability
3. Regularly rotate and audit access tokens
4. Use the provided cleanup scripts for safe container removal
5. Keep your MLflow and OIDC plugin versions up to date

---

*Last Updated: January 2025*
*Compatible with MLflow 2.x and OIDC authentication plugins*
