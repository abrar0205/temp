#!/usr/bin/env python3
"""
MLflow Token-Based Access Example Script

This script demonstrates how to:
1. Create and use access tokens with MLflow
2. Log experiments using authentication
3. Access experiment results using tokens

Prerequisites:
- MLflow server running with basic-auth enabled
- pip install mlflow requests
"""

import os
import sys
import requests
from requests.auth import HTTPBasicAuth

# MLflow tracking server URL
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "admin")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "admin123")


def create_user(username: str, password: str) -> dict:
    """
    Create a new user in MLflow using the admin credentials.
    
    Args:
        username: New user's username
        password: New user's password
    
    Returns:
        dict: API response
    """
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/users/create"
    
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
        return {"error": response.text}


def get_user(username: str) -> dict:
    """
    Get user information from MLflow.
    
    Args:
        username: Username to retrieve
    
    Returns:
        dict: User information
    """
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/users/get"
    
    response = requests.get(
        url,
        params={"username": username},
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get user: {response.text}")
        return {"error": response.text}


def list_experiments() -> dict:
    """
    List all experiments using token-based authentication.
    
    Returns:
        dict: List of experiments
    """
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search"
    
    response = requests.post(
        url,
        json={},
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to list experiments: {response.text}")
        return {"error": response.text}


def create_experiment(name: str) -> dict:
    """
    Create a new experiment using token-based authentication.
    
    Args:
        name: Name of the experiment
    
    Returns:
        dict: Created experiment details
    """
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/create"
    
    response = requests.post(
        url,
        json={"name": name},
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
    )
    
    if response.status_code == 200:
        print(f"✅ Experiment '{name}' created successfully")
        return response.json()
    else:
        print(f"❌ Failed to create experiment: {response.text}")
        return {"error": response.text}


def log_run_with_auth(experiment_id: str, run_name: str, metrics: dict, params: dict) -> dict:
    """
    Log a run with metrics and parameters using authentication.
    
    Args:
        experiment_id: ID of the experiment
        run_name: Name of the run
        metrics: Dictionary of metrics to log
        params: Dictionary of parameters to log
    
    Returns:
        dict: Run information
    """
    # Create a run
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/create"
    
    response = requests.post(
        url,
        json={
            "experiment_id": experiment_id,
            "run_name": run_name,
            "start_time": int(__import__('time').time() * 1000)
        },
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to create run: {response.text}")
        return {"error": response.text}
    
    run_data = response.json()
    run_id = run_data["run"]["info"]["run_id"]
    print(f"✅ Run '{run_name}' created with ID: {run_id}")
    
    # Log metrics
    for key, value in metrics.items():
        metric_url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/log-metric"
        requests.post(
            metric_url,
            json={
                "run_id": run_id,
                "key": key,
                "value": value,
                "timestamp": int(__import__('time').time() * 1000)
            },
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
        )
    print(f"✅ Logged {len(metrics)} metrics")
    
    # Log parameters
    for key, value in params.items():
        param_url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/log-parameter"
        requests.post(
            param_url,
            json={
                "run_id": run_id,
                "key": key,
                "value": str(value)
            },
            auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
        )
    print(f"✅ Logged {len(params)} parameters")
    
    return run_data


def get_experiment_results(experiment_id: str) -> dict:
    """
    Get experiment results (runs) using token-based authentication.
    
    Args:
        experiment_id: ID of the experiment
    
    Returns:
        dict: Runs in the experiment
    """
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search"
    
    response = requests.post(
        url,
        json={
            "experiment_ids": [experiment_id],
            "max_results": 100
        },
        auth=HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get experiment results: {response.text}")
        return {"error": response.text}


def verify_token_access():
    """
    Verify that token-based access works by performing all operations.
    """
    print("=" * 60)
    print("MLflow Token-Based Access Verification")
    print("=" * 60)
    print(f"\n📡 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"👤 Username: {MLFLOW_USERNAME}")
    print()
    
    # Test 1: List experiments
    print("\n🔍 Test 1: Listing experiments...")
    experiments = list_experiments()
    if "experiments" in experiments:
        print(f"   Found {len(experiments.get('experiments', []))} experiments")
    
    # Test 2: Create an experiment
    print("\n📝 Test 2: Creating a new experiment...")
    import time
    exp_name = f"token-test-experiment-{int(time.time())}"
    exp_result = create_experiment(exp_name)
    
    if "experiment_id" in exp_result:
        experiment_id = exp_result["experiment_id"]
        
        # Test 3: Log a run with metrics and parameters
        print("\n📊 Test 3: Logging a run with metrics and parameters...")
        metrics = {"accuracy": 0.95, "loss": 0.05, "f1_score": 0.93}
        params = {"learning_rate": 0.001, "epochs": 100, "batch_size": 32}
        run_result = log_run_with_auth(experiment_id, "test-run", metrics, params)
        
        # Test 4: Access experiment results
        print("\n📈 Test 4: Accessing experiment results...")
        results = get_experiment_results(experiment_id)
        if "runs" in results:
            print(f"   Found {len(results.get('runs', []))} runs in the experiment")
            for run in results.get("runs", []):
                print(f"   - Run ID: {run['info']['run_id']}")
                print(f"     Status: {run['info']['status']}")
                print(f"     Metrics: {run['data'].get('metrics', [])}")
    
    print("\n" + "=" * 60)
    print("✅ Token-based access verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    verify_token_access()
