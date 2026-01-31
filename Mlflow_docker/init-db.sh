#!/bin/bash
set -e

echo "Creating Keycloak database and user..."

# Create Keycloak user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
    CREATE ROLE ${KEYCLOAK_DB_USER} WITH LOGIN PASSWORD '${KEYCLOAK_DB_PASSWORD}';
EOSQL

# Create Keycloak database  
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<EOSQL
    CREATE DATABASE ${KEYCLOAK_DB_NAME} OWNER ${KEYCLOAK_DB_USER};
EOSQL

echo "Keycloak database setup complete!"
