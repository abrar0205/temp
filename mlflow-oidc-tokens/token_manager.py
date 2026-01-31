#!/usr/bin/env python3
"""
Keycloak Token Management Script

This script provides CLI tools to:
1. List active sessions/tokens for users
2. List all client sessions
3. Revoke/delete tokens

Since mlflow-oidc-auth does not provide UI for token management,
this script uses the Keycloak Admin API directly.

Prerequisites:
- Keycloak Admin credentials
- Python 3.8+
- requests package

Usage:
    export KEYCLOAK_ADMIN_PASSWORD="your-admin-password"
    export OIDC_CLIENT_SECRET="your-client-secret"
    
    # List all user sessions
    python token_manager.py list-sessions
    
    # List sessions for specific user
    python token_manager.py list-sessions --username mlflow_user
    
    # Revoke all sessions for a user
    python token_manager.py revoke-user --username mlflow_user
    
    # Show token management info
    python token_manager.py info
"""

import os
import sys
import argparse

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)


# Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
REALM = os.getenv("KEYCLOAK_REALM", "mlflow")
CLIENT_ID = os.getenv("OIDC_CLIENT_ID", "mlflow")
ADMIN_USERNAME = os.getenv("KEYCLOAK_ADMIN", "admin")
ADMIN_PASSWORD = os.getenv("KEYCLOAK_ADMIN_PASSWORD")
CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")


class KeycloakTokenManager:
    """Manage tokens via Keycloak Admin API."""
    
    def __init__(self, keycloak_url: str, realm: str, admin_username: str, admin_password: str):
        self.keycloak_url = keycloak_url
        self.realm = realm
        self.admin_username = admin_username
        self.admin_password = admin_password
    
    def _get_admin_token(self) -> str:
        """Get admin access token from master realm (fresh token each time to avoid expiration issues)."""
        response = requests.post(
            f"{self.keycloak_url}/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": self.admin_username,
                "password": self.admin_password,
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get admin token: {response.status_code} - {response.text}")
        
        return response.json()["access_token"]
    
    def _admin_headers(self) -> dict:
        """Get headers with admin token."""
        return {
            "Authorization": f"Bearer {self._get_admin_token()}",
            "Content-Type": "application/json",
        }
    
    def get_users(self, username: str = None) -> list:
        """Get users from realm, optionally filtered by username."""
        params = {}
        if username:
            params["username"] = username
        
        response = requests.get(
            f"{self.keycloak_url}/admin/realms/{self.realm}/users",
            headers=self._admin_headers(),
            params=params,
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get users: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_user_sessions(self, user_id: str) -> list:
        """Get active sessions for a specific user."""
        response = requests.get(
            f"{self.keycloak_url}/admin/realms/{self.realm}/users/{user_id}/sessions",
            headers=self._admin_headers(),
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get sessions: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_client_uuid(self, client_id: str) -> str:
        """Get internal UUID for a client."""
        response = requests.get(
            f"{self.keycloak_url}/admin/realms/{self.realm}/clients",
            headers=self._admin_headers(),
            params={"clientId": client_id},
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get clients: {response.status_code} - {response.text}")
        
        clients = response.json()
        if not clients:
            raise Exception(f"Client '{client_id}' not found")
        
        return clients[0]["id"]
    
    def get_client_sessions(self, client_id: str) -> list:
        """Get all sessions for a client."""
        client_uuid = self.get_client_uuid(client_id)
        
        response = requests.get(
            f"{self.keycloak_url}/admin/realms/{self.realm}/clients/{client_uuid}/user-sessions",
            headers=self._admin_headers(),
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get client sessions: {response.status_code} - {response.text}")
        
        return response.json()
    
    def logout_user(self, user_id: str) -> bool:
        """Logout all sessions for a user (revoke all tokens)."""
        response = requests.post(
            f"{self.keycloak_url}/admin/realms/{self.realm}/users/{user_id}/logout",
            headers=self._admin_headers(),
        )
        
        return response.status_code == 204
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session."""
        response = requests.delete(
            f"{self.keycloak_url}/admin/realms/{self.realm}/sessions/{session_id}",
            headers=self._admin_headers(),
        )
        
        return response.status_code == 204
    
    def get_realm_sessions_count(self) -> dict:
        """Get count of active sessions in realm."""
        response = requests.get(
            f"{self.keycloak_url}/admin/realms/{self.realm}/sessions/count",
            headers=self._admin_headers(),
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get session count: {response.status_code} - {response.text}")
        
        return response.json()


def format_session(session: dict) -> str:
    """Format a session for display."""
    user = session.get("username", "unknown")
    start = session.get("start", "unknown")
    last_access = session.get("lastAccess", "unknown")
    ip = session.get("ipAddress", "unknown")
    clients = session.get("clients", {})
    session_id = session.get("id", "unknown")
    
    return f"""  Session ID: {session_id}
    User: {user}
    Start: {start}
    Last Access: {last_access}
    IP Address: {ip}
    Clients: {', '.join(clients.keys())}"""


def cmd_list_sessions(args, manager: KeycloakTokenManager):
    """List active sessions."""
    print(f"\n{'='*60}")
    print("Active Sessions")
    print(f"{'='*60}")
    
    if args.username:
        # Get sessions for specific user
        users = manager.get_users(args.username)
        if not users:
            print(f"User '{args.username}' not found")
            return
        
        user = users[0]
        sessions = manager.get_user_sessions(user["id"])
        
        print(f"\nSessions for user '{args.username}':")
        if not sessions:
            print("  No active sessions")
        else:
            for session in sessions:
                print(format_session(session))
                print()
    else:
        # Get all client sessions
        try:
            sessions = manager.get_client_sessions(CLIENT_ID)
            print(f"\nAll sessions for client '{CLIENT_ID}':")
            if not sessions:
                print("  No active sessions")
            else:
                for session in sessions:
                    print(format_session(session))
                    print()
            
            # Show total count
            try:
                count = manager.get_realm_sessions_count()
                print(f"Total active sessions in realm: {count}")
            except Exception:
                pass
                
        except Exception as e:
            print(f"Error: {e}")


def cmd_revoke_user(args, manager: KeycloakTokenManager):
    """Revoke all sessions for a user."""
    if not args.username:
        print("Error: --username required")
        return
    
    users = manager.get_users(args.username)
    if not users:
        print(f"User '{args.username}' not found")
        return
    
    user = users[0]
    
    # Get current sessions before revocation
    sessions = manager.get_user_sessions(user["id"])
    
    print(f"\nRevoking {len(sessions)} session(s) for user '{args.username}'...")
    
    if manager.logout_user(user["id"]):
        print(f"✅ Successfully revoked all sessions for '{args.username}'")
    else:
        print(f"❌ Failed to revoke sessions for '{args.username}'")


def cmd_revoke_session(args, manager: KeycloakTokenManager):
    """Revoke a specific session."""
    if not args.session_id:
        print("Error: --session-id required")
        return
    
    print(f"\nRevoking session '{args.session_id}'...")
    
    if manager.delete_session(args.session_id):
        print(f"✅ Successfully revoked session")
    else:
        print(f"❌ Failed to revoke session")


def cmd_info(args, manager: KeycloakTokenManager):
    """Show token management information."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MLflow OIDC Token Management Information                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MLflow OIDC Auth Token Management:                                          ║
║  ──────────────────────────────────                                          ║
║                                                                              ║
║  As of mlflow-oidc-auth version 5.6.x:                                       ║
║  • No built-in UI for listing tokens                                         ║
║  • No built-in UI for deleting tokens                                        ║
║  • Token management is handled via Keycloak                                  ║
║                                                                              ║
║  Token Operations:                                                           ║
║  ─────────────────                                                           ║
║                                                                              ║
║  ┌─────────────────┬──────────────────────────────────────────────────────┐ ║
║  │ Operation       │ How                                                  │ ║
║  ├─────────────────┼──────────────────────────────────────────────────────┤ ║
║  │ Create Token    │ Keycloak token endpoint (password/client_creds)     │ ║
║  │ Use Token       │ Authorization: Bearer <token>                       │ ║
║  │ List Tokens     │ Keycloak Admin Console → Sessions                   │ ║
║  │                 │ OR: python token_manager.py list-sessions           │ ║
║  │ Delete Token    │ Keycloak Admin Console → Sessions → Sign out        │ ║
║  │                 │ OR: python token_manager.py revoke-user --username X│ ║
║  │ Token Expiry    │ Keycloak Admin → Realm Settings → Tokens            │ ║
║  └─────────────────┴──────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  Keycloak Admin Console: http://localhost:8080/admin                         ║
║  MLflow UI: http://localhost:5000                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Keycloak Token Manager for MLflow OIDC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all active sessions
    python token_manager.py list-sessions
    
    # List sessions for specific user
    python token_manager.py list-sessions --username mlflow_user
    
    # Revoke all sessions for a user
    python token_manager.py revoke-user --username mlflow_user
    
    # Revoke a specific session
    python token_manager.py revoke-session --session-id abc123
    
    # Show token management info
    python token_manager.py info

Environment Variables:
    KEYCLOAK_URL            Keycloak URL (default: http://localhost:8080)
    KEYCLOAK_REALM          Realm name (default: mlflow)
    KEYCLOAK_ADMIN          Admin username (default: admin)
    KEYCLOAK_ADMIN_PASSWORD Admin password (required)
    OIDC_CLIENT_ID          Client ID (default: mlflow)
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list-sessions command
    list_parser = subparsers.add_parser("list-sessions", help="List active sessions")
    list_parser.add_argument("--username", help="Filter by username")
    
    # revoke-user command
    revoke_user_parser = subparsers.add_parser("revoke-user", help="Revoke all sessions for a user")
    revoke_user_parser.add_argument("--username", required=True, help="Username")
    
    # revoke-session command
    revoke_session_parser = subparsers.add_parser("revoke-session", help="Revoke a specific session")
    revoke_session_parser.add_argument("--session-id", required=True, help="Session ID")
    
    # info command
    subparsers.add_parser("info", help="Show token management information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle info command without credentials
    if args.command == "info":
        cmd_info(args, None)
        return
    
    # Check admin password
    if not ADMIN_PASSWORD:
        print("Error: KEYCLOAK_ADMIN_PASSWORD environment variable not set")
        sys.exit(1)
    
    # Create manager
    try:
        manager = KeycloakTokenManager(
            keycloak_url=KEYCLOAK_URL,
            realm=REALM,
            admin_username=ADMIN_USERNAME,
            admin_password=ADMIN_PASSWORD,
        )
    except Exception as e:
        print(f"Error initializing Keycloak manager: {e}")
        sys.exit(1)
    
    # Execute command
    if args.command == "list-sessions":
        cmd_list_sessions(args, manager)
    elif args.command == "revoke-user":
        cmd_revoke_user(args, manager)
    elif args.command == "revoke-session":
        cmd_revoke_session(args, manager)


if __name__ == "__main__":
    main()
