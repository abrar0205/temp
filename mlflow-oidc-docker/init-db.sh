#!/bin/bash
# PostgreSQL initialization script
# Creates additional databases for Keycloak

set -e

echo "Creating additional PostgreSQL databases..."

# Create Keycloak database and user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create Keycloak database if it doesn't exist
    SELECT 'CREATE DATABASE ${KEYCLOAK_POSTGRES_DB:-keycloak}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${KEYCLOAK_POSTGRES_DB:-keycloak}')\gexec

    -- Create Keycloak user if it doesn't exist
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '${KEYCLOAK_POSTGRES_USER:-keycloak}') THEN
            CREATE USER ${KEYCLOAK_POSTGRES_USER:-keycloak} WITH PASSWORD '${KEYCLOAK_POSTGRES_PASSWORD:-keycloak_password}';
        END IF;
    END
    \$\$;

    -- Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE ${KEYCLOAK_POSTGRES_DB:-keycloak} TO ${KEYCLOAK_POSTGRES_USER:-keycloak};
EOSQL

# Connect to Keycloak database and set ownership
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "${KEYCLOAK_POSTGRES_DB:-keycloak}" <<-EOSQL
    -- Grant schema privileges to Keycloak user
    GRANT ALL ON SCHEMA public TO ${KEYCLOAK_POSTGRES_USER:-keycloak};
    ALTER SCHEMA public OWNER TO ${KEYCLOAK_POSTGRES_USER:-keycloak};
EOSQL

echo "PostgreSQL initialization completed successfully!"
echo "  - MLflow database: ${POSTGRES_DB:-mlflow}"
echo "  - Keycloak database: ${KEYCLOAK_POSTGRES_DB:-keycloak}"
