# MLflow OIDC Deployment - Troubleshooting Guide

This document details all errors encountered and solutions applied when deploying MLflow with the official `mlflow-oidc-auth` Docker image and Keycloak for OIDC authentication.


## Deployment Issues and Solutions

### 1. PostgreSQL Authentication Failure

**Error:**
```
FATAL: password authentication failed for user "mlflow"
```

**Root Cause:**
The `docker-compose.yml` had a hardcoded password `mlflow_password` in the MLflow command, but the `.env` file specified `mlflow123` as the PostgreSQL password.

**Solution:**
Updated the `--backend-store-uri` in the MLflow command to use the correct password:
```yaml
command: 
  - --backend-store-uri=postgresql://mlflow:mlflow123@postgres:5432/mlflow
```

**Lesson Learned:**
Always ensure database credentials match between environment variables and connection strings.

---

### 2. Keycloak Health Check Failure

**Error:**
```
/bin/sh: curl: not found
Health check failed
```

**Root Cause:**
The official Keycloak image (`quay.io/keycloak/keycloak:23.0`) doesn't include `curl` by default, causing the health check to fail.

**Solution:**
Changed health check from using `curl` to a TCP-based check:
```yaml
healthcheck:
  test: ["CMD-SHELL", "exec 3<>/dev/tcp/localhost/8080 && echo -e 'GET /health/ready HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n' >&3 && cat <&3 | grep -q 'HTTP/1.1 200' || exit 1"]
```

**Lesson Learned:**
Don't assume utilities are available in minimal Docker images. Use shell built-ins when possible.

---

### 3. Keycloak "HTTPS Required" Error

**Error:**
```
HTTPS required
```

**Root Cause:**
Keycloak's default configuration requires HTTPS for all external requests. In a local development environment without SSL certificates, this blocks all authentication attempts.

**Solution:**
Configured Keycloak to allow HTTP in development mode:
```yaml
environment:
  KC_HOSTNAME_STRICT_HTTPS: "false"
  KC_HTTP_ENABLED: "true"
command: start-dev --http-enabled=true --hostname-strict-https=false
```

Also updated the Keycloak database:
```sql
UPDATE realm SET ssl_required='NONE' WHERE name='mlflow';
```

**Lesson Learned:**
For local development, explicitly disable HTTPS requirements. For production, use proper SSL certificates.

---

### 4. Browser DNS Error: "keycloak's DNS address could not be found"

**Error:**
```
This site can't be reached
keycloak's DNS address could not be found
```

**Root Cause:**
Keycloak was generating redirect URLs using the internal Docker hostname `keycloak:8080`, which is only resolvable within the Docker network, not from the user's browser.

**Solution:**
Configured Keycloak to use `localhost` for browser-accessible URLs:
```yaml
environment:
  KC_HOSTNAME: localhost
  KC_HOSTNAME_PORT: "8080"
  KC_HOSTNAME_STRICT: "false"
```

While keeping the MLflow container's OIDC discovery URL pointing to the internal hostname:
```yaml
OIDC_DISCOVERY_URL: http://keycloak:8080/realms/mlflow/.well-known/openid-configuration
```

**Lesson Learned:**
Distinguish between:
- Internal container-to-container communication (use service names)
- External browser-to-container communication (use localhost or domain names)

---

### 5. MLflow Connection Refused to Keycloak

**Error:**
```
Failed to establish a new connection: [Errno 111] Connection refused
```

**Root Cause:**
The MLflow container was trying to connect to `localhost:8080`, which refers to its own container, not the Keycloak container.

**Solution:**
Ensured `OIDC_DISCOVERY_URL` uses the Docker service name for server-to-server communication:
```yaml
OIDC_DISCOVERY_URL: http://keycloak:8080/realms/mlflow/.well-known/openid-configuration
```

**Lesson Learned:**
Inside containers, `localhost` refers to the container itself. Always use Docker service names for inter-container communication.

---

### 6. Keycloak "Client not found" Error

**Error:**
```
Client not found
```

**Root Cause:**
The MLflow container was reading stale environment variables. The `.env` file had `OIDC_CLIENT_ID=mlflow-client`, but Keycloak was configured with `OIDC_CLIENT_ID=mlflow`.

**Solution:**
1. Updated `.env` file with correct client ID: `OIDC_CLIENT_ID=mlflow`
2. Force-recreated the MLflow container to pick up new environment variables:
```bash
docker compose up -d --force-recreate mlflow
```

**Lesson Learned:**
Docker containers don't automatically reload environment variables. Always force-recreate after `.env` changes.

---

### 7. Invalid OIDC Scopes Error

**Error:**
```
OIDC provider error: Invalid scopes: openid,email,profile
```

**Root Cause:**
This was the most persistent error. Investigation revealed:
1. The library was using comma-separated scopes (`openid,email,profile`) as a default
2. The environment variable was named `OIDC_SCOPES` (plural) in our config
3. The library actually expects `OIDC_SCOPE` (singular)
4. OIDC specification requires space-separated scopes, not comma-separated

**Investigation Process:**
```bash
# Found the source of the default value
docker exec mlflow-server grep -r "openid,email,profile" /mlflow/.venv/lib/python3.12/site-packages/mlflow_oidc_auth/

# Result:
# config.py: self.OIDC_SCOPE = os.environ.get("OIDC_SCOPE", "openid,email,profile")
```

**Solution:**
Changed the environment variable name from `OIDC_SCOPES` (plural) to `OIDC_SCOPE` (singular):
```yaml
environment:
  OIDC_SCOPE: "openid profile email"  # Space-separated, not comma-separated
```

**Lesson Learned:**
- Always check the library's source code for exact environment variable names
- OIDC standard uses space-separated scopes
- Singular vs. plural naming matters!

---

### 8. Session Error: "Missing OAuth state in session"

**Error:**
```
Session error: Missing OAuth state in session. Please try logging in again.
```

**Root Cause:**
The MLflow OIDC auth library defaults to `SESSION_TYPE=cachelib` (in-memory), which doesn't persist sessions properly. The Redis connection was configured via `REDIS_URL`, but the session module wasn't configured to use Redis.

**Investigation:**
```bash
# Found the session configuration
docker exec mlflow-server grep "SESSION_TYPE" /mlflow/.venv/lib/python3.12/site-packages/mlflow_oidc_auth/config.py

# Result: defaults to "cachelib"
```

**Solution:**
Explicitly configured Redis as the session backend:
```yaml
environment:
  SESSION_TYPE: redis
  REDIS_HOST: redis
  REDIS_PORT: "6379"
  REDIS_DB: "0"
```

**Lesson Learned:**
Setting `REDIS_URL` alone is not sufficient. The session module requires explicit configuration with separate host, port, and database variables.

---

### 9. Redis Connection Refused Error

**Error:**
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
```

**Root Cause:**
After setting `SESSION_TYPE: redis`, the library was trying to connect to `localhost:6379` instead of the `redis` service hostname.

**Investigation:**
Examined the Redis session module source:
```python
# mlflow_oidc_auth/session/redis.py
SESSION_REDIS = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),  # defaults to localhost!
    port=int(os.environ.get("REDIS_PORT", 6379)),
)
```

**Solution:**
Added explicit Redis connection parameters:
```yaml
environment:
  REDIS_HOST: redis      # Use Docker service name
  REDIS_PORT: "6379"
  REDIS_DB: "0"
```

**Lesson Learned:**
Libraries may have different default connection methods. Check the source code to understand what environment variables are actually used.

---

### 10. Group Detection Error

**Error:**
```
Group detection error: 'groups'
ERROR: Group detection error: Failed to get user groups
```

**Root Cause:**
The Keycloak ID token didn't include a `groups` claim because the default client scopes don't have a groups mapper configured.

**Solution:**
1. Created groups in Keycloak:
```bash
docker exec mlflow-keycloak /opt/keycloak/bin/kcadm.sh create groups -r mlflow -s name=mlflow-admins
```

2. Added a protocol mapper to include groups in the token:
```bash
docker exec mlflow-keycloak /opt/keycloak/bin/kcadm.sh create client-scopes/[SCOPE_ID]/protocol-mappers/models -r mlflow \
  -s name=groups \
  -s protocol=openid-connect \
  -s protocolMapper=oidc-group-membership-mapper \
  -s 'config."claim.name"=groups' \
  -s 'config."id.token.claim"=true' \
  -s 'config."access.token.claim"=true' \
  -s 'config."userinfo.token.claim"=true'
```

3. Added user to the group

**Lesson Learned:**
OIDC tokens only include claims that are explicitly mapped in the client configuration. Standard claims like `groups` must be added manually.

---

### 11. Authorization Error: "User is not allowed to login"

**Error:**
```
Authorization error: User is not allowed to login.
```

**Root Cause:**
The MLflow OIDC auth plugin checks if users belong to allowed groups. By default:
- `OIDC_ADMIN_GROUP_NAME` defaults to `mlflow-admin`
- `OIDC_GROUP_NAME` defaults to `mlflow`

Our group was named `mlflow_admins` (with underscore), which didn't match.

**Investigation:**
```bash
# Found the authorization check
docker exec mlflow-server grep "not allowed to login" /mlflow/.venv/lib/python3.12/site-packages/mlflow_oidc_auth/auth.py

# Found the default group names
docker exec mlflow-server grep "OIDC_ADMIN_GROUP_NAME\|OIDC_GROUP_NAME" /mlflow/.venv/lib/python3.12/site-packages/mlflow_oidc_auth/config.py
```

**Solution:**
Configured the environment variables to match our group names:
```yaml
environment:
  OIDC_ADMIN_GROUP_NAME: mlflow-admins
  OIDC_GROUP_NAME: mlflow-users
```

**Lesson Learned:**
Check library defaults and ensure your Keycloak configuration matches the expected group names, or override via environment variables.

---

## Key Takeaways

### Environment Variable Naming
- The library uses **singular** `OIDC_SCOPE`, not plural `OIDC_SCOPES`
- Redis session configuration requires: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` (not just `REDIS_URL`)
- Always verify exact variable names in the library source code

### Docker Networking
- Use service names for container-to-container communication
- Use `localhost` or domain names for browser access
- `KC_HOSTNAME` controls what Keycloak puts in redirect URLs (browser-facing)
- `OIDC_DISCOVERY_URL` should use service name (server-facing)

### Session Management
- Explicitly set `SESSION_TYPE: redis` to use Redis sessions
- In-memory sessions (`cachelib`) don't work for OAuth flows with redirects
- Always configure all required Redis connection parameters

### Keycloak Configuration
- Disable HTTPS requirements for local development
- Add group mappers to include groups in tokens
- Create groups before assigning users
- Realm SSL settings can override environment variables (check database)

### Debugging Strategies
1. Check container logs: `docker compose logs -f mlflow`
2. Inspect environment variables: `docker exec mlflow-server env | grep OIDC`
3. Read library source code in container: `docker exec mlflow-server cat /path/to/file.py`
4. Force-recreate containers after config changes: `docker compose up -d --force-recreate`
5. Clear Redis cache: `docker exec mlflow-redis redis-cli FLUSHALL`

### Production Recommendations
1. Use proper SSL certificates (remove `OAUTHLIB_INSECURE_TRANSPORT`)
2. Change all default passwords and secrets
3. Generate strong `SECRET_KEY` (32+ characters)
4. Use environment-specific `.env` files (never commit to git)
5. Configure Keycloak for production mode (not `start-dev`)
6. Set up proper health checks and monitoring
7. Configure database backups for PostgreSQL volumes

---

## Quick Reference: Working Configuration

### docker-compose.yml Key Settings
```yaml
keycloak:
  environment:
    KC_HOSTNAME: localhost              # For browser redirects
    KC_HOSTNAME_STRICT_HTTPS: "false"   # Allow HTTP in dev

mlflow:
  environment:
    OIDC_CLIENT_ID: mlflow
    OIDC_SCOPE: "openid profile email"  # Space-separated, SINGULAR
    OIDC_DISCOVERY_URL: http://keycloak:8080/realms/mlflow/.well-known/openid-configuration  # Use service name
    SESSION_TYPE: redis                 # Use Redis for sessions
    REDIS_HOST: redis                   # Service name
    REDIS_PORT: "6379"
    REDIS_DB: "0"
    OIDC_ADMIN_GROUP_NAME: mlflow-admins
    OIDC_GROUP_NAME: mlflow-users
```

### Required Keycloak Configuration
1. Realm: `mlflow`
2. Client ID: `mlflow`
3. Groups: `mlflow-admins`, `mlflow-users`
4. Client mapper: `groups` claim in ID token
5. SSL: `NONE` for development

---

## Additional Resources

- [MLflow OIDC Auth Documentation](https://github.com/mlflow-oidc/mlflow-oidc-auth)
- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [OIDC Specification](https://openid.net/specs/openid-connect-core-1_0.html)
- [Docker Compose Networking](https://docs.docker.com/compose/networking/)
