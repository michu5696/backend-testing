#!/usr/bin/env python3
"""
End-to-end testing script for the SSM (State-Space Model) pipeline.

This script tests the new backend-ssm API endpoints:
  1. Creates a project with training date range
  2. Creates conditioning/target feature sets
  3. Fetches historical data from FRED/Yahoo
  4. Creates a model linked to feature sets
  5. Trains an SSM model (GRU + Gaussian/Flow/Copula)
  6. Creates a scenario and runs a forecast
  7. Validates results

Key Differences from Old Pipeline:
- No sample/window generation - uses sequential time series
- State-space model (GRU/LSTM) instead of window clustering
- Generation models (Gaussian/Flow/Copula) instead of regime copulas
- Scenario-based conditioning (date range, metric filter) instead of sample picking

Requirements:
  * A running backend-ssm (e.g., `uvicorn main:app --reload` from backend-ssm/)
  * A valid admin API key
  * Network access to FRED/Yahoo for data ingestion

Example:
    # Full pipeline test (local backend)
    python test_ssm_pipeline.py --api-url http://localhost:8000 --stage all
    
    # Just training (existing model)
    python test_ssm_pipeline.py --stage training --model-id <uuid>
    
    # Just forecast (existing trained model)
    python test_ssm_pipeline.py --stage forecast --model-id <uuid>
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    plt = None
    mdates = None


# =============================================================================
# DEFAULT FEATURES (same as old pipeline for comparability)
# =============================================================================

DEFAULT_CONDITIONING_FEATURES: List[Dict[str, str]] = [
    {"id": "^IRX", "source": "yahoo", "name": "13-Week Treasury Yield"},
    {"id": "^FVX", "source": "yahoo", "name": "5-Year Treasury Yield"},
    {"id": "^TNX", "source": "yahoo", "name": "10-Year Treasury Yield"},
    {"id": "T10Y2Y", "source": "fred", "name": "10-Year Minus 2-Year Treasury Spread"},
    {"id": "T10YIE", "source": "fred", "name": "10-Year Breakeven Inflation Rate"},
    {"id": "BAMLC0A0CM", "source": "fred", "name": "Investment Grade Corporate Spread"},
    {"id": "VIXCLS", "source": "fred", "name": "VIX Volatility Index"},
]

DEFAULT_TARGET_FEATURES: List[Dict[str, str]] = [
    {"id": "SHY", "source": "yahoo", "name": "1-3 Year Treasury Bond ETF"},
    {"id": "IEI", "source": "yahoo", "name": "3-7 Year Treasury Bond ETF"},
    {"id": "IEF", "source": "yahoo", "name": "7-10 Year Treasury Bond ETF"},
    {"id": "TLT", "source": "yahoo", "name": "20+ Year Treasury Bond ETF"},
]


# =============================================================================
# UTILITIES
# =============================================================================

def load_default_api_key() -> Optional[str]:
    """Load API key from standard locations."""
    base_dir = Path(__file__).resolve().parents[1] / "backend" / "admin"
    for candidate in ["admin_api_key_v2.txt", "admin_api_key.txt"]:
        path = base_dir / candidate
        if path.exists():
            return path.read_text().strip()
    return None


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_step(step: str, message: str) -> None:
    """Print a step with formatting."""
    print(f"[{step}] {message}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"❌ {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"⚠️  {message}")


# =============================================================================
# API CLIENT
# =============================================================================

class SSMPipelineClient:
    """HTTP client for the SSM pipeline API."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        
        # Long timeouts for training operations
        timeout = httpx.Timeout(600.0, connect=30.0)
        self.session = httpx.Client(
            base_url=self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        
        self.user_id: Optional[str] = None
    
    def close(self) -> None:
        self.session.close()
    
    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request with error handling."""
        try:
            resp = self.session.request(method, path, **kwargs)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            raise RuntimeError(f"{method} {path} failed ({e.response.status_code}): {detail}")
        except httpx.TimeoutException:
            raise RuntimeError(f"{method} {path} timed out")
        
        if resp.content:
            return resp.json()
        return {}
    
    def get_user(self) -> Dict[str, Any]:
        """Get current user info."""
        data = self._request("GET", "/api/v1/account/user")
        self.user_id = data.get("user_id")
        return data
    
    # -------------------------------------------------------------------------
    # Project & Feature Set Management
    # -------------------------------------------------------------------------
    
    def list_projects(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all projects."""
        resp = self._request("GET", f"/api/v1/projects/?limit={limit}")
        return resp.get("projects", [])
    
    def create_project(
        self,
        name: str,
        description: str,
        training_start: str,
        training_end: str,
    ) -> Dict[str, Any]:
        """Create a new project."""
        payload = {
            "name": name,
            "description": description,
            "training_start_date": training_start,
            "training_end_date": training_end,
        }
        return self._request("POST", "/api/v1/projects/", json=payload)
    
    def get_or_create_project(
        self,
        name: str,
        description: str,
        training_start: str,
        training_end: str,
    ) -> Dict[str, Any]:
        """Get existing project by name or create new one."""
        projects = self.list_projects()
        for proj in projects:
            if proj.get("name") == name:
                print_step("project", f"Found existing project: {proj['id']}")
                return proj
        
        print_step("project", f"Creating new project: {name}")
        return self.create_project(name, description, training_start, training_end)
    
    def create_feature_set(
        self,
        project_id: str,
        name: str,
        description: str,
        features: List[Dict[str, str]],
        set_type: str,  # 'conditioning' or 'target'
        fred_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a feature set and add features to it."""
        # Step 1: Create the feature set
        create_payload = {
            "project_id": project_id,
            "name": name,
            "description": description,
            "set_type": set_type,
        }
        fs = self._request("POST", "/api/v1/feature-sets/", json=create_payload)
        
        # Step 2: Update with features and data collectors
        collectors = [{"source": "yahoo"}]
        if fred_api_key:
            collectors.append({"source": "fred", "api_key": fred_api_key})
        
        update_payload = {
            "features": features,
            "data_collectors": collectors,
        }
        return self._request("PATCH", f"/api/v1/feature-sets/{fs['id']}", json=update_payload)
    
    def get_feature_sets(self, project_id: str) -> List[Dict[str, Any]]:
        """Get feature sets for a project."""
        resp = self._request("GET", f"/api/v1/feature-sets/?project_id={project_id}")
        return resp.get("feature_sets", [])
    
    def get_or_create_feature_sets(
        self,
        project_id: str,
        conditioning_features: List[Dict[str, str]],
        target_features: List[Dict[str, str]],
        fred_api_key: Optional[str] = None,
    ) -> tuple:
        """Get or create conditioning and target feature sets."""
        existing = self.get_feature_sets(project_id)
        
        conditioning_set = None
        target_set = None
        
        for fs in existing:
            set_type = fs.get("set_type", "")
            name = fs.get("name", "").lower()
            if set_type == "conditioning" or "conditioning" in name:
                conditioning_set = fs
            elif set_type == "target" or "target" in name:
                target_set = fs
        
        if not conditioning_set:
            print_step("feature_set", "Creating conditioning feature set")
            conditioning_set = self.create_feature_set(
                project_id,
                "Conditioning Features",
                "Macro and market indicators",
                conditioning_features,
                "conditioning",
                fred_api_key,
            )
        else:
            print_step("feature_set", f"Using existing conditioning set: {conditioning_set['id']}")
        
        if not target_set:
            print_step("feature_set", "Creating target feature set")
            target_set = self.create_feature_set(
                project_id,
                "Target Features",
                "Assets to forecast",
                target_features,
                "target",
            )
        else:
            print_step("feature_set", f"Using existing target set: {target_set['id']}")
        
        return conditioning_set, target_set
    
    # -------------------------------------------------------------------------
    # Data Fetching
    # -------------------------------------------------------------------------
    
    def fetch_data(
        self,
        feature_set_id: str,
        start_date: str,
        end_date: str,
        fred_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch data for a feature set."""
        payload = {
            "start_date": start_date,
            "end_date": end_date,
        }
        if fred_api_key:
            payload["api_keys"] = {"fred": fred_api_key}
        
        # Use a long timeout client for data fetch
        fetch_timeout = httpx.Timeout(600.0, connect=60.0)
        with httpx.Client(
            base_url=self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=fetch_timeout,
        ) as fetch_client:
            resp = fetch_client.post(
                f"/api/v1/feature-sets/{feature_set_id}/fetch-data",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {}
    
    # -------------------------------------------------------------------------
    # Model Management
    # -------------------------------------------------------------------------
    
    def list_models(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models."""
        url = "/api/v1/models/"
        if project_id:
            url += f"?project_id={project_id}"
        resp = self._request("GET", url)
        return resp.get("models", [])
    
    def create_model(
        self,
        name: str,
        project_id: str,
        target_set_id: str,
        conditioning_set_id: str,
        training_start: str,
        training_end: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a model."""
        payload = {
            "name": name,
            "description": description,
            "project_id": project_id,
            "target_set_id": target_set_id,
            "conditioning_set_id": conditioning_set_id,
            "training_start_date": training_start,
            "training_end_date": training_end,
        }
        return self._request("POST", "/api/v1/models/", json=payload)
    
    def get_or_create_model(
        self,
        name: str,
        project_id: str,
        target_set_id: str,
        conditioning_set_id: str,
        training_start: str,
        training_end: str,
    ) -> Dict[str, Any]:
        """Get existing model by name or create new one."""
        models = self.list_models(project_id)
        for model in models:
            if model.get("name") == name:
                print_step("model", f"Found existing model: {model['id']}")
                return model
        
        print_step("model", f"Creating new model: {name}")
        return self.create_model(
            name, project_id, target_set_id, conditioning_set_id,
            training_start, training_end
        )
    
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    
    def train_model(
        self,
        model_id: str,
        state_model: str = "gru",
        generation_model: str = "gaussian",
        hidden_dim: Optional[int] = None,
        n_layers: int = 1,
        use_factors: bool = True,
        n_factors: Optional[int] = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Submit a training job."""
        payload = {
            "model_id": model_id,
            "state_model": state_model,
            "generation_model": generation_model,
            "n_layers": n_layers,
            "use_factors": use_factors,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
        }
        if hidden_dim:
            payload["hidden_dim"] = hidden_dim
        if n_factors:
            payload["n_factors"] = n_factors
        
        return self._request("POST", "/api/v1/ml/train", json=payload)
    
    def get_training_status(self, model_id: str) -> Dict[str, Any]:
        """Get training status."""
        return self._request("GET", f"/api/v1/ml/train/{model_id}/status")
    
    def poll_training(
        self,
        model_id: str,
        poll_interval: int = 10,
        timeout: int = 0,  # 0 = no timeout
    ) -> str:
        """Poll training status until complete."""
        elapsed = 0
        last_stage = None
        
        while timeout <= 0 or elapsed < timeout:
            try:
                status = self.get_training_status(model_id)
                current_status = status.get("status", "unknown")
                current_stage = status.get("current_stage", "")
                
                # Print status update
                if current_stage != last_stage:
                    print_step("train", f"Status: {current_status} | Stage: {current_stage} | Time: {elapsed}s")
                    last_stage = current_stage
                
                # Check for terminal states
                if current_status == "trained":
                    print_success(f"Training completed in {elapsed}s")
                    return "trained"
                elif current_status == "failed":
                    error = status.get("error", "Unknown error")
                    print_error(f"Training failed: {error}")
                    return "failed"
                
            except Exception as e:
                print_warning(f"Error polling status: {e}")
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        print_error(f"Training timed out after {elapsed}s")
        return "timeout"
    
    # -------------------------------------------------------------------------
    # Forecasting
    # -------------------------------------------------------------------------
    
    def forecast(
        self,
        model_id: str,
        scenario_name: str,
        horizon: int = 21,
        n_paths: int = 1000,
        feature_scenarios: Optional[Dict[str, Any]] = None,
        conditioning: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Run a forecast."""
        payload = {
            "model_id": model_id,
            "scenario_name": scenario_name,
            "horizon": horizon,
            "n_paths": n_paths,
        }
        if feature_scenarios:
            payload["feature_scenarios"] = feature_scenarios
        if conditioning:
            payload["conditioning"] = conditioning
        
        return self._request(
            "POST",
            "/api/v1/ml/forecast",
            json=payload,
            timeout=300.0,
        )
    
    def get_metric_distribution(
        self,
        feature: str,
        metric: str = "volatility",
        window_days: int = 21,
    ) -> Dict[str, Any]:
        """Get metric distribution for scenario exploration."""
        return self._request(
            "GET",
            f"/api/v1/ml/scenarios/metrics/{feature}",
            params={"metric": metric, "window_days": window_days},
        )


# =============================================================================
# TEST WORKFLOW
# =============================================================================

class SSMPipelineTest:
    """Test workflow for the SSM pipeline."""
    
    def __init__(
        self,
        client: SSMPipelineClient,
        project_name: str,
        training_start: str,
        training_end: str,
        conditioning_features: List[Dict[str, str]],
        target_features: List[Dict[str, str]],
        fred_api_key: Optional[str] = None,
        model_id_override: Optional[str] = None,
    ):
        self.client = client
        self.project_name = project_name
        self.training_start = training_start
        self.training_end = training_end
        self.conditioning_features = conditioning_features
        self.target_features = target_features
        self.fred_api_key = fred_api_key
        
        # State
        self.project: Optional[Dict[str, Any]] = None
        self.conditioning_set: Optional[Dict[str, Any]] = None
        self.target_set: Optional[Dict[str, Any]] = None
        self.model_id: Optional[str] = model_id_override
        self.model: Optional[Dict[str, Any]] = None
    
    def setup_infrastructure(self) -> None:
        """Set up project, feature sets, and model."""
        print_section("Setting Up Infrastructure")
        
        # Get user
        user = self.client.get_user()
        print_step("user", f"Authenticated as: {user.get('email', user.get('user_id'))}")
        
        # Project
        self.project = self.client.get_or_create_project(
            self.project_name,
            "SSM Pipeline Test Project",
            self.training_start,
            self.training_end,
        )
        print_step("project", f"Project ID: {self.project['id']}")
        
        # Feature sets
        self.conditioning_set, self.target_set = self.client.get_or_create_feature_sets(
            self.project["id"],
            self.conditioning_features,
            self.target_features,
            self.fred_api_key,
        )
        print_step("feature_set", f"Conditioning: {self.conditioning_set['id']}")
        print_step("feature_set", f"Target: {self.target_set['id']}")
    
    def fetch_data(self) -> None:
        """Fetch training data for feature sets."""
        print_section("Fetching Training Data")
        
        # Fetch conditioning data
        print_step("fetch", "Fetching conditioning data...")
        result = self.client.fetch_data(
            self.conditioning_set["id"],
            self.training_start,
            self.training_end,
            self.fred_api_key,
        )
        print_step("fetch", f"Conditioning: {result.get('features_fetched', 0)} features, {result.get('total_processed_points', 0)} points")
        
        # Fetch target data
        print_step("fetch", "Fetching target data...")
        result = self.client.fetch_data(
            self.target_set["id"],
            self.training_start,
            self.training_end,
        )
        print_step("fetch", f"Target: {result.get('features_fetched', 0)} features, {result.get('total_processed_points', 0)} points")
        
        print_success("Data fetched successfully")
    
    def create_model(self, model_name: str) -> None:
        """Create or get model."""
        print_section("Creating Model")
        
        if self.model_id:
            print_step("model", f"Using provided model ID: {self.model_id}")
            return
        
        self.model = self.client.get_or_create_model(
            model_name,
            self.project["id"],
            self.target_set["id"],
            self.conditioning_set["id"],
            self.training_start,
            self.training_end,
        )
        self.model_id = self.model["id"]
        print_step("model", f"Model ID: {self.model_id}")
    
    def train_model(
        self,
        state_model: str = "gru",
        generation_model: str = "gaussian",
        max_epochs: int = 50,
        poll_interval: int = 10,
        timeout: int = 0,
    ) -> str:
        """Train the model."""
        print_section(f"Training Model ({state_model} + {generation_model})")
        
        if not self.model_id:
            raise RuntimeError("Model must be created before training")
        
        # Submit training
        print_step("train", "Submitting training job...")
        result = self.client.train_model(
            self.model_id,
            state_model=state_model,
            generation_model=generation_model,
            max_epochs=max_epochs,
        )
        print_step("train", f"Job submitted: status={result.get('status')}")
        
        # If training runs synchronously (local dev), it's already done
        if result.get("status") == "trained":
            print_success("Training completed (synchronous)")
            return "trained"
        
        # Poll for async training
        print_step("train", "Polling for completion...")
        return self.client.poll_training(self.model_id, poll_interval, timeout)
    
    def run_forecast_unconditional(
        self,
        horizon: int = 21,
        n_paths: int = 500,
    ) -> Dict[str, Any]:
        """Run unconditional forecast."""
        print_section("Running Unconditional Forecast")
        
        if not self.model_id:
            raise RuntimeError("Model must be trained before forecasting")
        
        print_step("forecast", f"Horizon: {horizon} days, Paths: {n_paths}")
        result = self.client.forecast(
            self.model_id,
            "Unconditional Forecast",
            horizon=horizon,
            n_paths=n_paths,
        )
        
        print_step("forecast", f"Status: {result.get('scenario_type', 'unknown')}")
        print_step("forecast", f"Simulation ID: {result.get('simulation_id')}")
        
        # Print summary
        summary = result.get("summary", [])
        if summary:
            print_step("forecast", "Summary statistics:")
            for stat in summary[:3]:  # First 3 features
                print(f"    {stat['feature']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")
        
        print_success("Unconditional forecast completed")
        return result
    
    def run_forecast_with_scenario(
        self,
        scenario_name: str,
        feature: str,
        start_date: str,
        end_date: str,
        horizon: int = 21,
        n_paths: int = 500,
    ) -> Dict[str, Any]:
        """Run forecast with date-range scenario."""
        print_section(f"Running Scenario Forecast: {scenario_name}")
        
        if not self.model_id:
            raise RuntimeError("Model must be trained before forecasting")
        
        print_step("forecast", f"Conditioning {feature} on {start_date} to {end_date}")
        
        feature_scenarios = {
            feature: {
                "mode": "date_range",
                "start_date": start_date,
                "end_date": end_date,
            }
        }
        
        result = self.client.forecast(
            self.model_id,
            scenario_name,
            horizon=horizon,
            n_paths=n_paths,
            feature_scenarios=feature_scenarios,
        )
        
        print_step("forecast", f"Scenario type: {result.get('scenario_type')}")
        print_step("forecast", f"Simulation ID: {result.get('simulation_id')}")
        
        print_success(f"Scenario forecast '{scenario_name}' completed")
        return result
    
    def run_full_pipeline(
        self,
        model_name: str,
        state_model: str = "gru",
        generation_model: str = "gaussian",
        max_epochs: int = 50,
        poll_interval: int = 10,
        training_timeout: int = 0,
        forecast_horizon: int = 21,
        n_paths: int = 500,
    ) -> Dict[str, Any]:
        """Run the full pipeline end-to-end."""
        results = {
            "setup": False,
            "data_fetch": False,
            "model_creation": False,
            "training": False,
            "forecast_unconditional": False,
            "forecast_scenario": False,
        }
        
        try:
            # Setup
            self.setup_infrastructure()
            results["setup"] = True
            
            # Data
            self.fetch_data()
            results["data_fetch"] = True
            
            # Model
            self.create_model(model_name)
            results["model_creation"] = True
            
            # Training
            status = self.train_model(
                state_model=state_model,
                generation_model=generation_model,
                max_epochs=max_epochs,
                poll_interval=poll_interval,
                timeout=training_timeout,
            )
            results["training"] = (status == "trained")
            
            if not results["training"]:
                print_error("Training failed, skipping forecasts")
                return results
            
            # Forecasts
            self.run_forecast_unconditional(
                horizon=forecast_horizon,
                n_paths=n_paths,
            )
            results["forecast_unconditional"] = True
            
            # Scenario forecast (COVID crash)
            self.run_forecast_with_scenario(
                "COVID Crash Scenario",
                "VIX Volatility Index",
                "2020-03-01",
                "2020-03-21",
                horizon=forecast_horizon,
                n_paths=n_paths,
            )
            results["forecast_scenario"] = True
            
        except Exception as e:
            print_error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SSM Pipeline")
    
    # API settings
    parser.add_argument(
        "--api-url",
        default=os.environ.get("SABLIER_API_URL", "http://localhost:8000"),
        help="Backend API URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SABLIER_API_KEY") or load_default_api_key(),
        help="API key",
    )
    parser.add_argument(
        "--fred-api-key",
        default=os.environ.get("FRED_API_KEY", "3d349d18e62cab7c2d55f1a6680f06d8"),
        help="FRED API key",
    )
    
    # Test configuration
    parser.add_argument(
        "--stage",
        choices=["all", "setup", "training", "forecast"],
        default="all",
        help="Which stage to run",
    )
    parser.add_argument("--model-id", help="Use existing model ID")
    parser.add_argument("--project-name", default="SSM Pipeline Test")
    parser.add_argument("--model-name", default="SSM Test Model")
    parser.add_argument("--training-start", default="2015-01-01")
    parser.add_argument("--training-end", default="2024-12-01")
    
    # Model architecture
    parser.add_argument(
        "--state-model",
        choices=["gru", "lstm"],
        default="gru",
        help="State model type",
    )
    parser.add_argument(
        "--generation-model",
        choices=["gaussian", "flow", "copula"],
        default="gaussian",
        help="Generation model type",
    )
    
    # Training settings
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--training-timeout", type=int, default=0, help="0 = no timeout")
    
    # Forecast settings
    parser.add_argument("--forecast-horizon", type=int, default=21)
    parser.add_argument("--n-paths", type=int, default=500)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.api_key:
        print_error("API key not provided. Use --api-key or set SABLIER_API_KEY.")
        sys.exit(1)
    
    print_section("SSM Pipeline Test")
    print(f"API URL: {args.api_url}")
    print(f"Stage: {args.stage}")
    print(f"State Model: {args.state_model}")
    print(f"Generation Model: {args.generation_model}")
    
    client = SSMPipelineClient(args.api_url, args.api_key)
    
    try:
        test = SSMPipelineTest(
            client,
            args.project_name,
            args.training_start,
            args.training_end,
            DEFAULT_CONDITIONING_FEATURES,
            DEFAULT_TARGET_FEATURES,
            args.fred_api_key,
            args.model_id,
        )
        
        if args.stage == "all":
            results = test.run_full_pipeline(
                args.model_name,
                state_model=args.state_model,
                generation_model=args.generation_model,
                max_epochs=args.max_epochs,
                poll_interval=args.poll_interval,
                training_timeout=args.training_timeout,
                forecast_horizon=args.forecast_horizon,
                n_paths=args.n_paths,
            )
            
            # Print summary
            print_section("Test Results")
            for stage, success in results.items():
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {stage}: {status}")
            
            all_passed = all(results.values())
            if all_passed:
                print_success("All tests passed!")
            else:
                print_error("Some tests failed")
                sys.exit(1)
        
        elif args.stage == "setup":
            test.setup_infrastructure()
            test.fetch_data()
            test.create_model(args.model_name)
            print_success("Setup complete")
        
        elif args.stage == "training":
            if not args.model_id:
                # Need to set up first
                test.setup_infrastructure()
                test.fetch_data()
                test.create_model(args.model_name)
            
            status = test.train_model(
                state_model=args.state_model,
                generation_model=args.generation_model,
                max_epochs=args.max_epochs,
                poll_interval=args.poll_interval,
                timeout=args.training_timeout,
            )
            
            if status != "trained":
                sys.exit(1)
        
        elif args.stage == "forecast":
            if not args.model_id:
                print_error("--model-id required for forecast stage")
                sys.exit(1)
            
            test.run_forecast_unconditional(
                horizon=args.forecast_horizon,
                n_paths=args.n_paths,
            )
    
    finally:
        client.close()


if __name__ == "__main__":
    main()

