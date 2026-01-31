#!/usr/bin/env python3
"""
test_mlflow_token_access.py - Test MLflow access with OIDC tokens

This script verifies:
1. Getting an access token from Keycloak
2. Using the token to access MLflow experiments
3. Creating and logging to experiments using tokens
"""

import os
import sys
import json
import requests
from datetime import datetime

# Configuration
KEYCLOAK_URL = os.environ.get("KEYCLOAK_URL", "http://localhost:8080")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")
KEYCLOAK_REALM = os.environ.get("KEYCLOAK_REALM", "mlflow")
OIDC_CLIENT_ID = os.environ.get("OIDC_CLIENT_ID", "mlflow")
OIDC_CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "")
TEST_USERNAME = os.environ.get("TEST_USERNAME", "mlflow_user")
TEST_PASSWORD = os.environ.get("TEST_PASSWORD", "password123")


def get_access_token():
    """Get an access token from Keycloak using password grant."""
    token_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
    
    data = {
        "grant_type": "password",
        "client_id": OIDC_CLIENT_ID,
        "client_secret": OIDC_CLIENT_SECRET,
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD,
        "scope": "openid profile email"
    }
    
    try:
        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token")
        else:
            print(f"    ✗ Token request failed: {response.status_code}")
            print(f"    Response: {response.text}")
            return None
    except Exception as e:
        print(f"    ✗ Error getting token: {e}")
        return None


def test_mlflow_health():
    """Test MLflow server health."""
    print("\n[1] Testing MLflow server health...")
    try:
        response = requests.get(f"{MLFLOW_URL}/health", timeout=10)
        if response.status_code == 200:
            print(f"    ✓ MLflow server is healthy")
            return True
        else:
            print(f"    ? MLflow returned status: {response.status_code}")
            return True  # Server is responding
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_token_acquisition():
    """Test getting an access token from Keycloak."""
    print("\n[2] Testing access token acquisition from Keycloak...")
    token = get_access_token()
    if token:
        print(f"    ✓ Successfully obtained access token")
        print(f"    Token preview: {token[:50]}...")
        return token
    else:
        print(f"    ✗ Failed to get access token")
        return None


def test_mlflow_api_with_token(token):
    """Test accessing MLflow API with the access token."""
    print("\n[3] Testing MLflow API access with token...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Try to list experiments
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            headers=headers,
            json={"max_results": 100},
            timeout=10
        )
        
        if response.status_code == 200:
            experiments = response.json().get("experiments", [])
            print(f"    ✓ Successfully accessed experiments: found {len(experiments)} experiment(s)")
            return True
        elif response.status_code == 401:
            print(f"    ✗ Authentication failed (401)")
            print(f"    Response: {response.text[:200]}")
            return False
        else:
            print(f"    ? Unexpected status: {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_create_experiment_with_token(token):
    """Test creating an experiment using access token."""
    print("\n[4] Testing experiment creation with token...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    experiment_name = f"token-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/create",
            headers=headers,
            json={"name": experiment_name},
            timeout=10
        )
        
        if response.status_code == 200:
            experiment_id = response.json().get("experiment_id")
            print(f"    ✓ Created experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id
        else:
            print(f"    ✗ Failed to create experiment: {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def test_log_run_with_token(token, experiment_id):
    """Test logging a run using access token."""
    print("\n[5] Testing run logging with token...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Create a run
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/create",
            headers=headers,
            json={"experiment_id": experiment_id},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"    ✗ Failed to create run: {response.status_code}")
            return False
        
        run_id = response.json()["run"]["info"]["run_id"]
        print(f"    ✓ Created run with ID: {run_id}")
        
        # Log a parameter
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/log-parameter",
            headers=headers,
            json={"run_id": run_id, "key": "test_param", "value": "test_value"},
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Logged parameter: test_param = test_value")
        
        # Log a metric
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/log-metric",
            headers=headers,
            json={"run_id": run_id, "key": "accuracy", "value": 0.95, "timestamp": int(datetime.now().timestamp() * 1000)},
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Logged metric: accuracy = 0.95")
        
        # End the run
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/update",
            headers=headers,
            json={"run_id": run_id, "status": "FINISHED"},
            timeout=10
        )
        if response.status_code == 200:
            print(f"    ✓ Run completed successfully")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_retrieve_results_with_token(token, experiment_id):
    """Test retrieving experiment results using token."""
    print("\n[6] Testing experiment results retrieval with token...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            headers=headers,
            json={"experiment_ids": [experiment_id], "max_results": 10},
            timeout=10
        )
        
        if response.status_code == 200:
            runs = response.json().get("runs", [])
            print(f"    ✓ Retrieved {len(runs)} run(s) from experiment {experiment_id}")
            
            for i, run in enumerate(runs, 1):
                run_id = run["info"]["run_id"]
                status = run["info"]["status"]
                metrics = run.get("data", {}).get("metrics", [])
                params = run.get("data", {}).get("params", [])
                
                print(f"\n    Run {i}:")
                print(f"      ID: {run_id}")
                print(f"      Status: {status}")
                if metrics:
                    print(f"      Metrics: {', '.join([f'{m[\"key\"]}={m[\"value\"]}' for m in metrics])}")
                if params:
                    print(f"      Params: {', '.join([f'{p[\"key\"]}={p[\"value\"]}' for p in params])}")
            
            return True
        else:
            print(f"    ✗ Failed to retrieve runs: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("MLflow OIDC Token Access Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Keycloak URL: {KEYCLOAK_URL}")
    print(f"  MLflow URL:   {MLFLOW_URL}")
    print(f"  Client ID:    {OIDC_CLIENT_ID}")
    print(f"  Test User:    {TEST_USERNAME}")
    
    all_passed = True
    
    # Test 1: MLflow health
    if not test_mlflow_health():
        print("\n✗ MLflow server is not accessible. Exiting.")
        sys.exit(1)
    
    # Test 2: Token acquisition
    token = test_token_acquisition()
    if not token:
        print("\n✗ Could not obtain access token. Check Keycloak configuration.")
        print("Make sure you have run ./configure-keycloak.sh and updated OIDC_CLIENT_SECRET")
        sys.exit(1)
    
    # Test 3: API access with token
    if not test_mlflow_api_with_token(token):
        print("\n✗ API access with token failed.")
        all_passed = False
    
    # Test 4: Create experiment
    experiment_id = test_create_experiment_with_token(token)
    if not experiment_id:
        print("\n✗ Could not create experiment.")
        all_passed = False
    else:
        # Test 5: Log run
        if not test_log_run_with_token(token, experiment_id):
            all_passed = False
        
        # Test 6: Retrieve results
        if not test_retrieve_results_with_token(token, experiment_id):
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Token-based access is working correctly.")
    else:
        print("✗ Some tests failed. Check the output above.")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
