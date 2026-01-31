#!/usr/bin/env python3
"""
MLflow Access Token Management Script

This script provides functionality to:
1. Create access tokens for users
2. List existing access tokens (via database query - not available in standard MLflow API)
3. Delete/revoke access tokens

Note: MLflow's standard basic-auth does not provide a UI or API for listing/deleting tokens.
This script demonstrates workarounds using direct database access.

Prerequisites:
- MLflow server running with basic-auth enabled
- pip install mlflow requests sqlite3
"""

import os
import sys
import sqlite3
import argparse
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "admin")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "admin123")

# Path to the basic_auth.db (used when running locally)
# In Docker, this would be mounted at /mlflow/basic_auth.db
AUTH_DB_PATH = os.getenv("MLFLOW_AUTH_DB_PATH", "./basic_auth.db")


class TokenManager:
    """
    Manager class for MLflow access tokens.
    
    Note on MLflow Token Management:
    ================================
    As of MLflow 2.10.x, the standard MLflow basic-auth plugin does NOT provide:
    - UI for listing tokens
    - UI for deleting tokens
    - API endpoints for listing all tokens
    - API endpoints for deleting tokens
    
    Token management must be done through:
    1. Direct database manipulation (SQLite for basic-auth)
    2. Custom API implementations
    3. Third-party solutions like mlflow-oidc-auth
    
    The mlflow-oidc-auth project (https://github.com/mlflow/mlflow-oidc-auth) 
    provides enhanced authentication features including better token management,
    but as of version 0.1.x, full token listing/deletion UI is still limited.
    """
    
    def __init__(self, db_path: str = None, tracking_uri: str = None):
        self.db_path = db_path or AUTH_DB_PATH
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        
    def create_user(self, username: str, password: str) -> dict:
        """Create a new user via MLflow API."""
        url = f"{self.tracking_uri}/api/2.0/mlflow/users/create"
        
        response = requests.post(
            url,
            json={"username": username, "password": password},
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
        )
        
        if response.status_code == 200:
            print(f"✅ User '{username}' created successfully")
            return response.json()
        else:
            print(f"❌ Failed to create user: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
    
    def update_user_password(self, username: str, new_password: str) -> dict:
        """Update a user's password via MLflow API."""
        url = f"{self.tracking_uri}/api/2.0/mlflow/users/update-password"
        
        response = requests.patch(
            url,
            json={"username": username, "password": new_password},
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
        )
        
        if response.status_code == 200:
            print(f"✅ Password updated for user '{username}'")
            return {"success": True}
        else:
            print(f"❌ Failed to update password: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
    
    def delete_user(self, username: str) -> dict:
        """Delete a user via MLflow API."""
        url = f"{self.tracking_uri}/api/2.0/mlflow/users/delete"
        
        response = requests.delete(
            url,
            json={"username": username},
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
        )
        
        if response.status_code == 200:
            print(f"✅ User '{username}' deleted successfully")
            return {"success": True}
        else:
            print(f"❌ Failed to delete user: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
    
    def get_user(self, username: str) -> dict:
        """Get user information via MLflow API."""
        url = f"{self.tracking_uri}/api/2.0/mlflow/users/get"
        
        response = requests.get(
            url,
            params={"username": username},
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get user: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
    
    def list_users_from_db(self) -> list:
        """
        List all users directly from the SQLite database.
        
        Note: This requires access to the basic_auth.db file.
        In a Docker environment, you need to mount this file or exec into the container.
        
        Returns:
            list: List of user dictionaries
        """
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found at: {self.db_path}")
            print("   If running in Docker, use: docker exec -it mlflow-server sqlite3 /mlflow/basic_auth.db")
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the users table
            cursor.execute("""
                SELECT id, username, is_admin, experiment_permissions, registered_model_permissions
                FROM users
            """)
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    "id": row[0],
                    "username": row[1],
                    "is_admin": bool(row[2]),
                    "experiment_permissions": row[3],
                    "registered_model_permissions": row[4]
                })
            
            conn.close()
            return users
            
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
            return []
    
    def list_tokens_from_db(self) -> list:
        """
        List all access tokens directly from the SQLite database.
        
        Note: MLflow basic-auth does not store separate API tokens - 
        it uses username/password for authentication.
        
        For OIDC-based setups, tokens are typically managed by the identity provider.
        
        Returns:
            list: Empty list for basic-auth (no separate tokens table)
        """
        print("ℹ️  MLflow basic-auth uses username/password authentication.")
        print("   There is no separate tokens table in basic_auth.db.")
        print("   For token-based access, use the credentials directly.")
        print("")
        print("   For OIDC authentication with separate token management,")
        print("   consider using mlflow-oidc-auth which provides:")
        print("   - OAuth2/OIDC integration")
        print("   - Token-based authentication")
        print("   - Better token lifecycle management")
        return []
    
    def revoke_user_access(self, username: str) -> bool:
        """
        Revoke a user's access by deleting them.
        
        In MLflow basic-auth, revoking access means deleting the user
        or changing their password.
        
        Args:
            username: Username to revoke
            
        Returns:
            bool: True if successful
        """
        result = self.delete_user(username)
        return "error" not in result


def print_token_management_info():
    """Print information about token management in MLflow."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MLflow Token Management Information                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MLflow Authentication Options:                                              ║
║  ─────────────────────────────                                               ║
║                                                                              ║
║  1. Basic Auth (--app-name basic-auth)                                       ║
║     • Uses username/password stored in SQLite                                ║
║     • No separate API tokens                                                 ║
║     • Credentials ARE the tokens                                             ║
║     • No UI for token listing/deletion (password change = token rotation)    ║
║                                                                              ║
║  2. OIDC Auth (mlflow-oidc-auth plugin)                                      ║
║     • Integrates with OAuth2/OIDC providers (Keycloak, Okta, etc.)          ║
║     • Tokens managed by Identity Provider                                    ║
║     • Token listing/deletion through IdP admin console                       ║
║     • GitHub: https://github.com/mlflow/mlflow-oidc-auth                    ║
║                                                                              ║
║  Token Operations Available:                                                 ║
║  ──────────────────────────                                                  ║
║                                                                              ║
║  ┌─────────────────┬──────────────┬──────────────┬─────────────────────────┐║
║  │ Operation       │ Basic Auth   │ OIDC Auth    │ How                     │║
║  ├─────────────────┼──────────────┼──────────────┼─────────────────────────┤║
║  │ Create Token    │ Create User  │ IdP Login    │ API/CLI/UI              │║
║  │ Use Token       │ HTTP Basic   │ Bearer Token │ Auth Header             │║
║  │ List Tokens     │ Query DB     │ IdP Console  │ sqlite3/Admin Console   │║
║  │ Delete Token    │ Delete User  │ IdP Console  │ API/Admin Console       │║
║  │ Rotate Token    │ Change Pass  │ IdP Refresh  │ API/Admin Console       │║
║  └─────────────────┴──────────────┴──────────────┴─────────────────────────┘║
║                                                                              ║
║  For full token management UI, consider:                                     ║
║  • Keycloak + mlflow-oidc-auth                                              ║
║  • Custom authentication proxy                                               ║
║  • Enterprise MLflow solutions                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="MLflow Access Token Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new user (which serves as their token)
  python token_manager.py create-user --username testuser --password testpass123
  
  # Get user information
  python token_manager.py get-user --username testuser
  
  # Update user password (rotate token)
  python token_manager.py update-password --username testuser --password newpass456
  
  # Delete user (revoke token)
  python token_manager.py delete-user --username testuser
  
  # List all users from database
  python token_manager.py list-users --db-path ./basic_auth.db
  
  # Show token management information
  python token_manager.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create user command
    create_parser = subparsers.add_parser("create-user", help="Create a new user")
    create_parser.add_argument("--username", required=True, help="Username")
    create_parser.add_argument("--password", required=True, help="Password")
    
    # Get user command
    get_parser = subparsers.add_parser("get-user", help="Get user information")
    get_parser.add_argument("--username", required=True, help="Username")
    
    # Update password command
    update_parser = subparsers.add_parser("update-password", help="Update user password")
    update_parser.add_argument("--username", required=True, help="Username")
    update_parser.add_argument("--password", required=True, help="New password")
    
    # Delete user command
    delete_parser = subparsers.add_parser("delete-user", help="Delete a user")
    delete_parser.add_argument("--username", required=True, help="Username")
    
    # List users command
    list_parser = subparsers.add_parser("list-users", help="List all users from database")
    list_parser.add_argument("--db-path", default=AUTH_DB_PATH, help="Path to basic_auth.db")
    
    # Info command
    subparsers.add_parser("info", help="Show token management information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = TokenManager()
    
    if args.command == "create-user":
        manager.create_user(args.username, args.password)
    
    elif args.command == "get-user":
        user = manager.get_user(args.username)
        if "user" in user:
            print(f"\nUser Information:")
            print(f"  Username: {user['user'].get('username')}")
            print(f"  Is Admin: {user['user'].get('is_admin')}")
    
    elif args.command == "update-password":
        manager.update_user_password(args.username, args.password)
    
    elif args.command == "delete-user":
        manager.delete_user(args.username)
    
    elif args.command == "list-users":
        manager.db_path = args.db_path
        users = manager.list_users_from_db()
        if users:
            print(f"\nFound {len(users)} users:")
            for user in users:
                admin_badge = " [ADMIN]" if user["is_admin"] else ""
                print(f"  - {user['username']}{admin_badge}")
    
    elif args.command == "info":
        print_token_management_info()


if __name__ == "__main__":
    main()
