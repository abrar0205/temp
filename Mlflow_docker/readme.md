# MLflow with Keycloak OIDC Authentication

MLflow deployment with Keycloak SSO authentication.

## Architecture

- **MLflow Server**: Tracking server with OIDC authentication
- **Keycloak**: OpenID Connect provider for SSO
- **PostgreSQL**: Backend storage for MLflow and Keycloak
- **Redis**: Session storage
- **MinIO**: S3-compatible artifact storage

## Building the Docker Image

The MLflow OIDC server image is built from the official [mlflow-tracking-server-docker] https://github.com/mlflow-oidc/mlflow-tracking-server-docker
### Build Command

```bash
cd mlflow-tracking-server-docker
docker build --build-arg UV_HTTP_TIMEOUT=300 -t registry.gitlab.com/morris-darren.babu/mlflow-oidc-server:latest .
```

### Command Explanation

- `docker build` - Build a Docker image from a Dockerfile
- `--build-arg UV_HTTP_TIMEOUT=300` - Set timeout for the `uv` package manager to 300 seconds
  - `uv` is a fast Python package installer (alternative to pip)
  - The timeout prevents build failures when downloading large packages (mlflow, dependencies)
  - 300 seconds (5 minutes) allows sufficient time for slow network connections
- `-t registry.gitlab.com/morris-darren.babu/mlflow-oidc-server:latest` - Tag the image
  - `registry.gitlab.com/morris-darren.babu/mlflow-oidc-server` - Image name in GitLab Container Registry
  - `latest` - Version tag (can be changed to specific versions like `3.6.0-5.6.1`)
- `.` - Build context (current directory contains the Dockerfile)

### What Gets Installed

The Dockerfile installs:
- Python 3.12
- MLflow 3.6.0
- mlflow-oidc-auth 5.6.1 (OIDC authentication plugin)
- psycopg2 (PostgreSQL driver)
- boto3 (S3/MinIO client)
- redis (Redis client for sessions)

### Push to Registry

After building, push the image to GitLab Container Registry:

```bash
docker push registry.gitlab.com/morris-darren.babu/mlflow-oidc-server:latest
```

**Note:** You need to be logged in to the GitLab Container Registry:
```bash
docker login registry.gitlab.com
```

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Generate a secure secret key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Edit .env and update:
# - SECRET_KEY (use generated value above)
# - All passwords (POSTGRES_PASSWORD, KEYCLOAK_ADMIN_PASSWORD, etc.)
```

### 2. Start Services

```bash
docker compose up -d
```

### 3. Configure Keycloak

```bash
# Wait for all services to be healthy (takes ~60 seconds)
docker compose ps

# Run configuration script to set up realm, client, and test user
./configure-keycloak.sh
```

The script will output the `OIDC_CLIENT_SECRET`. Update your `.env` file with this value and restart MLflow:

```bash
# Update OIDC_CLIENT_SECRET in .env file
# Then restart MLflow
docker compose restart mlflow
```

### 4. Access MLflow

- **MLflow UI**: http://localhost:5000
- **Keycloak Admin**: http://localhost:8080
- **MinIO Console**: http://localhost:9001

**Default test user** (created by configure-keycloak.sh):
- Username: `mlflow_user`
- Password: `password123`

## Configuration

### Environment Variables

See `.env.example` for all options. Required variables:

- `SECRET_KEY`: Session encryption key (32+ random characters)
- `OIDC_CLIENT_SECRET`: From Keycloak after running configure-keycloak.sh
- All database passwords

### User Groups

Users must belong to one of these groups (configured in Keycloak):

- `mlflow_admins`: Full administrative access
- `mlflow_users`: Regular user access

Update group names via `OIDC_ADMIN_GROUP_NAME` and `OIDC_GROUP_NAME` in `.env`.

## Management

### View Logs
```bash
docker compose logs -f mlflow
docker compose logs -f keycloak
```

### Stop Services
```bash
docker compose down
```

### Reset All Data
```bash
docker compose down -v  # WARNING: Deletes all data
```

### Add New Users

1. Access Keycloak admin console: http://localhost:8080
2. Login with admin credentials from `.env`
3. Navigate to MLflow realm → Users → Add User
4. Set password in Credentials tab
5. Add user to `mlflow_admins` or `mlflow_users` group in Groups tab

## Troubleshooting

### "Invalid scopes" error
- Ensure `.env` has `OIDC_SCOPE` (singular, not SCOPES)
- Value should be: `"openid profile email"`

### "Session error: Missing OAuth state"
- Check Redis is running: `docker compose ps redis`
- Verify Redis connection: `docker exec mlflow-redis redis-cli ping`

### "Authorization error: User is not allowed to login"
- User must be in `mlflow_admins` or `mlflow_users` group in Keycloak
- Check group membership in Keycloak admin console
- Verify `OIDC_ADMIN_GROUP_NAME` and `OIDC_GROUP_NAME` in `.env` match Keycloak groups

### "Internal Server Error"
- Check MLflow logs: `docker compose logs mlflow`
- Verify all environment variables are set correctly in `.env`

## Files

- `docker-compose.yml`: Service definitions
- `.env`: Environment configuration (DO NOT COMMIT)
- `.env.example`: Template for .env file
- `init-db.sh`: PostgreSQL initialization script
- `configure-keycloak.sh`: Keycloak setup automation script
- `.gitignore`: Excludes sensitive files from git

## Support

For issues with:
- MLflow OIDC: https://github.com/mlflow-oidc/mlflow-oidc-auth
- Keycloak: https://www.keycloak.org/documentation
