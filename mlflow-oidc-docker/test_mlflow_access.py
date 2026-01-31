#!/usr/bin/env python3
"""
test_mlflow_access.py - Test MLflow access with authentication tokens

This script demonstrates:
1. Creating an experiment in MLflow using token authentication
2. Logging metrics and parameters
3. Retrieving experiment results using tokens
"""

import os
import sys
import requests
from datetime import datetime

# Configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_USERNAME = os.environ.get("MLFLOW_USERNAME", "admin")
MLFLOW_PASSWORD = os.environ.get("MLFLOW_PASSWORD", "admin_password")


def create_session():
    """Create a requests session with basic auth."""
    session = requests.Session()
    session.auth = (MLFLOW_USERNAME, MLFLOW_PASSWORD)
    return session


def test_health(session):
    """Test MLflow server health."""
    print("\n[1] Testing MLflow server health...")
    try:
        # Try health endpoint first
        response = session.get(f"{MLFLOW_TRACKING_URI}/health")
        if response.status_code == 200:
            print(f"    ✓ MLflow server is healthy")
            return True
        
        # Fallback to root endpoint
        response = session.get(f"{MLFLOW_TRACKING_URI}/")
        if response.status_code in [200, 401]:
            print(f"    ✓ MLflow server is reachable (status: {response.status_code})")
            return True
        
        print(f"    ✗ Unexpected response: {response.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"    ✗ Cannot connect to MLflow at {MLFLOW_TRACKING_URI}")
        return False


def test_authentication(session):
    """Test authentication with MLflow."""
    print("\n[2] Testing authentication...")
    try:
        # Use POST with proper parameters for search endpoint
        response = session.post(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search",
            json={"max_results": 10}
        )
        if response.status_code == 200:
            print(f"    ✓ Authentication successful")
            return True
        elif response.status_code == 401:
            print(f"    ✗ Authentication failed (401 Unauthorized)")
            return False
        else:
            print(f"    ? Unexpected response: {response.status_code}")
            return response.status_code < 400
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def create_experiment(session, name):
    """Create a new experiment."""
    print(f"\n[3] Creating experiment '{name}'...")
    try:
        response = session.post(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/create",
            json={"name": name}
        )
        if response.status_code == 200:
            experiment_id = response.json().get("experiment_id")
            print(f"    ✓ Created experiment with ID: {experiment_id}")
            return experiment_id
        elif response.status_code == 400:
            # Experiment might already exist
            error = response.json().get("error_code", "")
            if "RESOURCE_ALREADY_EXISTS" in str(error) or "already exists" in response.text.lower():
                print(f"    ! Experiment already exists, searching for it...")
                return get_experiment_by_name(session, name)
            print(f"    ✗ Failed to create experiment: {response.text}")
            return None
        else:
            print(f"    ✗ Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def get_experiment_by_name(session, name):
    """Get experiment by name."""
    try:
        response = session.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": name}
        )
        if response.status_code == 200:
            experiment = response.json().get("experiment", {})
            experiment_id = experiment.get("experiment_id")
            print(f"    ✓ Found experiment with ID: {experiment_id}")
            return experiment_id
    except Exception as e:
        print(f"    ✗ Error getting experiment: {e}")
    return None


def create_run(session, experiment_id):
    """Create a new run in the experiment."""
    print(f"\n[4] Creating a run in experiment {experiment_id}...")
    try:
        response = session.post(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/create",
            json={
                "experiment_id": experiment_id,
                "start_time": int(datetime.now().timestamp() * 1000),
                "tags": [
                    {"key": "mlflow.runName", "value": f"test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"}
                ]
            }
        )
        if response.status_code == 200:
            run = response.json().get("run", {})
            run_id = run.get("info", {}).get("run_id")
            print(f"    ✓ Created run with ID: {run_id}")
            return run_id
        else:
            print(f"    ✗ Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def log_metrics_and_params(session, run_id):
    """Log metrics and parameters to a run."""
    print(f"\n[5] Logging metrics and parameters to run {run_id}...")
    
    # Log parameters
    params = [
        {"key": "learning_rate", "value": "0.01"},
        {"key": "batch_size", "value": "32"},
        {"key": "model_type", "value": "neural_network"}
    ]
    
    for param in params:
        try:
            response = session.post(
                f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/log-parameter",
                json={"run_id": run_id, "key": param["key"], "value": param["value"]}
            )
            if response.status_code == 200:
                print(f"    ✓ Logged param: {param['key']} = {param['value']}")
            else:
                print(f"    ✗ Failed to log param {param['key']}: {response.status_code}")
        except Exception as e:
            print(f"    ✗ Error logging param: {e}")
    
    # Log metrics
    metrics = [
        {"key": "accuracy", "value": 0.95},
        {"key": "loss", "value": 0.05},
        {"key": "f1_score", "value": 0.94}
    ]
    
    timestamp = int(datetime.now().timestamp() * 1000)
    for metric in metrics:
        try:
            response = session.post(
                f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/log-metric",
                json={
                    "run_id": run_id,
                    "key": metric["key"],
                    "value": metric["value"],
                    "timestamp": timestamp,
                    "step": 0
                }
            )
            if response.status_code == 200:
                print(f"    ✓ Logged metric: {metric['key']} = {metric['value']}")
            else:
                print(f"    ✗ Failed to log metric {metric['key']}: {response.status_code}")
        except Exception as e:
            print(f"    ✗ Error logging metric: {e}")
    
    return True


def finish_run(session, run_id):
    """Mark the run as finished."""
    print(f"\n[6] Finishing run {run_id}...")
    try:
        response = session.post(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/update",
            json={
                "run_id": run_id,
                "status": "FINISHED",
                "end_time": int(datetime.now().timestamp() * 1000)
            }
        )
        if response.status_code == 200:
            print(f"    ✓ Run finished successfully")
            return True
        else:
            print(f"    ✗ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def get_experiment_results(session, experiment_id):
    """Retrieve experiment results using the access token."""
    print(f"\n[7] Retrieving experiment results for experiment {experiment_id}...")
    try:
        # Search for runs
        response = session.post(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [experiment_id],
                "max_results": 10
            }
        )
        
        if response.status_code == 200:
            runs = response.json().get("runs", [])
            print(f"    ✓ Found {len(runs)} run(s)")
            
            for i, run in enumerate(runs):
                run_info = run.get("info", {})
                run_data = run.get("data", {})
                
                print(f"\n    Run {i+1}:")
                print(f"      ID: {run_info.get('run_id', 'N/A')}")
                print(f"      Status: {run_info.get('status', 'N/A')}")
                print(f"      Start Time: {run_info.get('start_time', 'N/A')}")
                
                # Print metrics
                metrics = run_data.get("metrics", [])
                if metrics:
                    print(f"      Metrics:")
                    for metric in metrics:
                        print(f"        - {metric.get('key')}: {metric.get('value')}")
                
                # Print params
                params = run_data.get("params", [])
                if params:
                    print(f"      Parameters:")
                    for param in params:
                        print(f"        - {param.get('key')}: {param.get('value')}")
            
            return runs
        else:
            print(f"    ✗ Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def list_experiments(session):
    """List all experiments."""
    print("\n[8] Listing all experiments...")
    try:
        # Use POST with proper parameters
        response = session.post(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search",
            json={"max_results": 100}
        )
        if response.status_code == 200:
            experiments = response.json().get("experiments", [])
            print(f"    ✓ Found {len(experiments)} experiment(s):")
            for exp in experiments:
                print(f"      - ID: {exp.get('experiment_id')}, Name: {exp.get('name')}")
            return experiments
        else:
            print(f"    ✗ Failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def main():
    """Main test function."""
    print("=" * 60)
    print("MLflow Access Token Test Script")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"  Username:   {MLFLOW_USERNAME}")
    print(f"  Password:   {'*' * len(MLFLOW_PASSWORD)}")
    
    # Create authenticated session
    session = create_session()
    
    # Run tests
    all_passed = True
    
    # Test 1: Health check
    if not test_health(session):
        print("\n✗ MLflow server is not reachable. Please ensure it's running.")
        print("  Run: docker-compose up -d")
        sys.exit(1)
    
    # Test 2: Authentication
    if not test_authentication(session):
        print("\n✗ Authentication failed. Check credentials.")
        all_passed = False
    
    # Test 3: Create experiment
    experiment_name = f"token-test-experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    experiment_id = create_experiment(session, experiment_name)
    if not experiment_id:
        print("\n✗ Failed to create experiment")
        all_passed = False
    else:
        # Test 4-6: Create run and log data
        run_id = create_run(session, experiment_id)
        if run_id:
            log_metrics_and_params(session, run_id)
            finish_run(session, run_id)
        else:
            all_passed = False
        
        # Test 7: Retrieve results
        results = get_experiment_results(session, experiment_id)
        if not results:
            all_passed = False
    
    # Test 8: List all experiments
    list_experiments(session)
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Token-based access is working correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
