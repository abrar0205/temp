#!/usr/bin/env python3
"""
MLflow OIDC Token Access Example

This script demonstrates how to:
1. Obtain access tokens from Keycloak
2. Use tokens to access MLflow experiments
3. Log metrics and parameters with token authentication

Prerequisites:
- MLflow OIDC deployment running (see Mlflow_docker/)
- Keycloak configured with 'mlflow' realm and 'mlflow' client
- User account with mlflow_users or mlflow_admins group membership

Usage:
    export OIDC_CLIENT_SECRET="your-client-secret"
    python token_access_example.py
"""

import os
import sys
import json
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

try:
    import mlflow
except ImportError:
    print("Error: 'mlflow' package required. Install with: pip install mlflow")
    sys.exit(1)


# Configuration - Update these for your environment
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
REALM = os.getenv("KEYCLOAK_REALM", "mlflow")
CLIENT_ID = os.getenv("OIDC_CLIENT_ID", "mlflow")
CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")
USERNAME = os.getenv("MLFLOW_USERNAME", "mlflow_user")
PASSWORD = os.getenv("MLFLOW_PASSWORD", "password123")


def get_access_token(username: str, password: str) -> dict:
    """
    Get access token from Keycloak using password grant.
    
    Returns:
        dict with access_token, refresh_token, expires_in, etc.
    """
    if not CLIENT_SECRET:
        raise ValueError("OIDC_CLIENT_SECRET environment variable not set")
    
    token_url = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
    
    response = requests.post(
        token_url,
        data={
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "username": username,
            "password": password,
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to get token: {response.status_code} - {response.text}")
    
    return response.json()


def refresh_access_token(refresh_token: str) -> dict:
    """
    Refresh an access token using the refresh token.
    
    Args:
        refresh_token: The refresh token from initial authentication
        
    Returns:
        dict with new access_token, refresh_token, etc.
    """
    token_url = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
    
    response = requests.post(
        token_url,
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token,
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to refresh token: {response.status_code} - {response.text}")
    
    return response.json()


def revoke_token(token: str) -> bool:
    """
    Revoke an access or refresh token.
    
    Args:
        token: The token to revoke
        
    Returns:
        True if successful
    """
    revoke_url = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/revoke"
    
    response = requests.post(
        revoke_url,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "token": token,
        }
    )
    
    return response.status_code == 200


def list_experiments_with_api(access_token: str) -> list:
    """
    List experiments using MLflow REST API with token authentication.
    
    Args:
        access_token: Bearer token for authentication
        
    Returns:
        List of experiments
    """
    response = requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={}
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to list experiments: {response.status_code} - {response.text}")
    
    return response.json().get("experiments", [])


def list_experiments_with_sdk(access_token: str) -> list:
    """
    List experiments using MLflow Python SDK with token authentication.
    
    Args:
        access_token: Bearer token for authentication
        
    Returns:
        List of experiments
    """
    # Set token for SDK
    os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
    mlflow.set_tracking_uri(MLFLOW_URL)
    
    return mlflow.search_experiments()


def log_experiment_run(access_token: str, experiment_name: str = None):
    """
    Create an experiment and log a run with metrics/parameters.
    
    Args:
        access_token: Bearer token for authentication
        experiment_name: Name of experiment (generated if not provided)
    """
    # Set token for SDK
    os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
    mlflow.set_tracking_uri(MLFLOW_URL)
    
    # Create or get experiment
    if experiment_name is None:
        experiment_name = f"token-test-{int(time.time())}"
    
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
    
    # Log a run
    with mlflow.start_run(experiment_id=experiment_id, run_name="test-run"):
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 10)
        
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("loss", 0.05)
        mlflow.log_metric("f1_score", 0.93)
        
        mlflow.set_tag("model_type", "neural_network")
        mlflow.set_tag("created_by", "token_access_example.py")
        
        print("Logged run with metrics and parameters")
    
    return experiment_id


def get_experiment_results(access_token: str, experiment_id: str) -> dict:
    """
    Get runs from an experiment using token authentication.
    
    Args:
        access_token: Bearer token for authentication
        experiment_id: ID of the experiment
        
    Returns:
        Dict with runs and their metrics
    """
    response = requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "experiment_ids": [experiment_id],
            "max_results": 100,
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to get runs: {response.status_code} - {response.text}")
    
    return response.json()


def main():
    print("=" * 60)
    print("MLflow OIDC Token Access Demonstration")
    print("=" * 60)
    
    # Check required environment variable
    if not CLIENT_SECRET:
        print("\n❌ Error: OIDC_CLIENT_SECRET environment variable not set")
        print("   Set it with: export OIDC_CLIENT_SECRET='your-secret'")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Keycloak URL: {KEYCLOAK_URL}")
    print(f"  MLflow URL: {MLFLOW_URL}")
    print(f"  Realm: {REALM}")
    print(f"  Client ID: {CLIENT_ID}")
    print(f"  Username: {USERNAME}")
    
    # Step 1: Get access token
    print("\n" + "-" * 40)
    print("Step 1: Obtaining access token from Keycloak")
    try:
        tokens = get_access_token(USERNAME, PASSWORD)
        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token")
        expires_in = tokens.get("expires_in", "unknown")
        
        print(f"  ✅ Access token obtained")
        print(f"     Expires in: {expires_in} seconds")
        print(f"     Token preview: {access_token[:50]}...")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        sys.exit(1)
    
    # Step 2: List experiments using REST API
    print("\n" + "-" * 40)
    print("Step 2: Listing experiments via REST API")
    try:
        experiments = list_experiments_with_api(access_token)
        print(f"  ✅ Found {len(experiments)} experiment(s)")
        for exp in experiments[:5]:  # Show first 5
            print(f"     - {exp.get('name', 'unnamed')} (ID: {exp.get('experiment_id')})")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Step 3: List experiments using SDK
    print("\n" + "-" * 40)
    print("Step 3: Listing experiments via Python SDK")
    try:
        experiments = list_experiments_with_sdk(access_token)
        print(f"  ✅ Found {len(experiments)} experiment(s)")
        for exp in experiments[:5]:
            print(f"     - {exp.name} (ID: {exp.experiment_id})")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Step 4: Create experiment and log run
    print("\n" + "-" * 40)
    print("Step 4: Creating experiment and logging run")
    try:
        experiment_id = log_experiment_run(access_token)
        print(f"  ✅ Experiment created and run logged")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        experiment_id = None
    
    # Step 5: Get experiment results
    if experiment_id:
        print("\n" + "-" * 40)
        print("Step 5: Retrieving experiment results")
        try:
            results = get_experiment_results(access_token, experiment_id)
            runs = results.get("runs", [])
            print(f"  ✅ Found {len(runs)} run(s)")
            for run in runs:
                info = run.get("info", {})
                data = run.get("data", {})
                metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
                print(f"     Run ID: {info.get('run_id', 'unknown')[:8]}...")
                print(f"     Status: {info.get('status', 'unknown')}")
                print(f"     Metrics: {metrics}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Step 6: Demonstrate token refresh
    if refresh_token:
        print("\n" + "-" * 40)
        print("Step 6: Refreshing access token")
        try:
            new_tokens = refresh_access_token(refresh_token)
            new_access_token = new_tokens["access_token"]
            print(f"  ✅ Token refreshed successfully")
            print(f"     New token preview: {new_access_token[:50]}...")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Step 7: Demonstrate token revocation
    print("\n" + "-" * 40)
    print("Step 7: Token revocation (demonstration)")
    print("  ℹ️  Skipping actual revocation to keep token valid")
    print("     To revoke, call: revoke_token(access_token)")
    
    print("\n" + "=" * 60)
    print("✅ Token access demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
