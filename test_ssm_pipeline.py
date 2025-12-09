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
    import pandas as pd
except ImportError:
    plt = None
    mdates = None
    pd = None


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
    project_root = Path(__file__).resolve().parents[1]
    
    # Check backend-ssm first (for v3)
    backend_ssm_dir = project_root / "backend-ssm" / "admin"
    for candidate in ["admin_api_key_v3.txt", "admin_api_key.txt"]:
        path = backend_ssm_dir / candidate
        if path.exists():
            return path.read_text().strip()
    
    # Fallback to backend (for v2)
    backend_dir = project_root / "backend" / "admin"
    for candidate in ["admin_api_key_v2.txt", "admin_api_key.txt"]:
        path = backend_dir / candidate
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


def _plot_forecast_paths(
    forecast_result: Dict[str, Any],
    feature_name: str,
    output_path: Path,
    n_paths_to_plot: int = 100,
    scenario_name: str = "Unconditional",
) -> None:
    """
    Plot forecast paths with confidence intervals over time and individual paths.
    
    Args:
        forecast_result: Forecast API response with 'summary' and 'dates'
        feature_name: Which feature to plot
        output_path: Where to save the plot
        n_paths_to_plot: Number of individual paths to plot
        scenario_name: Name for the plot title
    """
    print(f"[PLOT] Starting plot for feature: {feature_name}")
    
    if plt is None or pd is None:
        print_warning("matplotlib or pandas not available; skipping plot")
        return
    
    summary = forecast_result.get('summary', [])
    dates_raw = forecast_result.get('dates', [])
    
    if not summary or not dates_raw:
        print_warning("No summary data or dates to plot")
        return
    
    # Find data for the requested feature
    feature_data = None
    for item in summary:
        if item.get('feature') == feature_name:
            feature_data = item
            break
    
    if not feature_data:
        print_warning(f"Feature '{feature_name}' not found in summary")
        return
    
    # Extract time-series statistics
    mean_vals = feature_data.get('mean', [])
    std_vals = feature_data.get('std', [])
    q05 = feature_data.get('q05', [])
    q25 = feature_data.get('q25', [])
    q50 = feature_data.get('q50', [])
    q75 = feature_data.get('q75', [])
    q95 = feature_data.get('q95', [])
    
    if not mean_vals or not isinstance(mean_vals, list):
        print_warning(f"No time-series mean values to plot (got {type(mean_vals)})")
        return
    
    # Parse dates
    try:
        dates = pd.to_datetime(dates_raw)
    except:
        dates = np.arange(len(mean_vals))
    
    # Get historical context if available
    historical_context = forecast_result.get('historical_context', {})
    import logging
    logger = logging.getLogger(__name__)
    
    print(f"[DEBUG] Historical context keys: {list(historical_context.keys()) if historical_context else 'None'}")
    print(f"[DEBUG] Looking for feature: {feature_name}")
    
    hist_data = None
    hist_dates = None
    hist_dates_raw = None
    
    # Try exact match first
    if historical_context and feature_name in historical_context:
        hist_feat = historical_context[feature_name]
        print(f"[DEBUG] Found historical data for {feature_name}: type={type(hist_feat)}")
        
        if isinstance(hist_feat, dict) and 'values' in hist_feat and 'dates' in hist_feat:
            hist_data = hist_feat['values']
            hist_dates_raw = hist_feat['dates']
            print(f"[DEBUG] Historical data: {len(hist_data)} values, {len(hist_dates_raw)} dates")
            try:
                hist_dates = pd.to_datetime(hist_dates_raw)
                print(f"[DEBUG] Successfully parsed {len(hist_dates)} historical dates")
            except Exception as e:
                print(f"[WARNING] Failed to parse historical dates: {e}")
                hist_dates = None
        elif isinstance(hist_feat, list):
            hist_data = hist_feat
            hist_dates = None
            hist_dates_raw = None
            print(f"[DEBUG] Historical data as list: {len(hist_data)} values")
    else:
        # Try case-insensitive or partial match
        if historical_context:
            for key in historical_context.keys():
                if key.lower() == feature_name.lower() or feature_name.lower() in key.lower():
                    print(f"[DEBUG] Found approximate match: '{key}' for '{feature_name}'")
                    hist_feat = historical_context[key]
                    if isinstance(hist_feat, dict) and 'values' in hist_feat and 'dates' in hist_feat:
                        hist_data = hist_feat['values']
                        hist_dates_raw = hist_feat['dates']
                        try:
                            hist_dates = pd.to_datetime(hist_dates_raw)
                        except:
                            hist_dates = None
                    break
        
        if not hist_data:
            print(f"[WARNING] No historical context found for {feature_name}. Available keys: {list(historical_context.keys()) if historical_context else 'None'}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot historical data if available - ALWAYS plot if we have data
    has_history = False
    if hist_data and len(hist_data) > 0:
        print(f"[PLOT] Attempting to plot {len(hist_data)} historical data points")
        
        # Strategy: Always try to plot with dates, fall back if needed
        plotted = False
        
        # Try 1: Use parsed dates if they match
        if hist_dates is not None and len(hist_data) == len(hist_dates):
            try:
                ax.plot(hist_dates, hist_data, 'k-', linewidth=2.5, label='Historical', alpha=0.8, zorder=3)
                has_history = True
                plotted = True
                print(f"[PLOT] ✓ Plotted {len(hist_data)} historical points with parsed dates")
            except Exception as e:
                print(f"[PLOT] Failed to plot with parsed dates: {e}")
        
        # Try 2: Generate date range from forecast dates
        if not plotted and dates_raw and len(dates_raw) > 0:
            try:
                forecast_dates = pd.to_datetime(dates_raw)
                first_forecast_date = forecast_dates[0]
                # Generate historical dates going back from first forecast date
                hist_date_range = pd.date_range(end=first_forecast_date, periods=len(hist_data), freq='B')  # Business days
                ax.plot(hist_date_range, hist_data, 'k-', linewidth=2.5, label='Historical', alpha=0.8, zorder=3)
                has_history = True
                plotted = True
                print(f"[PLOT] ✓ Plotted {len(hist_data)} historical points with generated date range")
            except Exception as e:
                print(f"[PLOT] Failed to generate date range: {e}")
        
        # Try 3: Use raw date strings if available
        if not plotted and hist_dates_raw and len(hist_dates_raw) == len(hist_data):
            try:
                hist_dates_parsed = pd.to_datetime(hist_dates_raw)
                ax.plot(hist_dates_parsed, hist_data, 'k-', linewidth=2.5, label='Historical', alpha=0.8, zorder=3)
                has_history = True
                plotted = True
                print(f"[PLOT] ✓ Plotted {len(hist_data)} historical points from raw date strings")
            except Exception as e:
                print(f"[PLOT] Failed to parse raw dates: {e}")
        
        # Add vertical line to separate history from forecast
        if has_history and dates_raw and len(dates_raw) > 0:
            try:
                first_forecast_date = pd.to_datetime(dates_raw[0])
                ax.axvline(x=first_forecast_date, color='black', linestyle='--', alpha=0.6, linewidth=2, zorder=2, label='Forecast Start')
                print(f"[PLOT] ✓ Added separator line at {first_forecast_date}")
            except Exception as e:
                print(f"[PLOT] Could not add separator line: {e}")
        
        if not plotted:
            print(f"[WARNING] Historical data exists ({len(hist_data)} points) but could not be plotted with any method!")
        
        if has_history:
            print_step("plot", f"Plotted {len(hist_data)} historical data points")
    
    # Plot individual paths (if available)
    paths_data = forecast_result.get('paths', {})
    if paths_data and feature_name in paths_data:
        paths_array = np.array(paths_data[feature_name])  # (n_paths, horizon)
        n_to_plot = min(n_paths_to_plot, paths_array.shape[0])
        for i in range(n_to_plot):
            ax.plot(dates, paths_array[i], color='darkgray', alpha=0.15, linewidth=0.8, zorder=1)
        print_step("plot", f"Plotted {n_to_plot} individual paths")
    else:
        print_warning(f"No paths data found. paths_data keys: {list(paths_data.keys()) if paths_data else 'None'}")
    
    # Plot confidence intervals
    if q05 and q95 and len(q05) == len(dates):
        ax.fill_between(dates, q05, q95, alpha=0.2, color='blue', label='5%-95%', zorder=2)
    if q25 and q75 and len(q25) == len(dates):
        ax.fill_between(dates, q25, q75, alpha=0.3, color='blue', label='25%-75%', zorder=2)
    
    # Plot median and mean
    if q50 and len(q50) == len(dates):
        ax.plot(dates, q50, 'b-', linewidth=2.5, label='Median', alpha=0.9, zorder=4)
    if mean_vals and len(mean_vals) == len(dates):
        ax.plot(dates, mean_vals, 'r--', linewidth=2.5, label='Mean', alpha=0.9, zorder=4)
    
    # Formatting
    ax.set_title(f"{feature_name} - {scenario_name} Forecast", fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis if using dates
    if isinstance(dates, pd.DatetimeIndex) and mdates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=30)
    
    # Add stats box
    final_mean = mean_vals[-1] if isinstance(mean_vals, list) else mean_vals
    final_std = std_vals[-1] if isinstance(std_vals, list) else std_vals
    stats_text = f"Final: {final_mean:.2f} ± {final_std:.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_step("plot", f"Saved forecast preview to {output_path}")


# =============================================================================
# API CLIENT
# =============================================================================

class SSMPipelineClient:
    """HTTP client for the SSM pipeline API."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        
        # Detect if local (synchronous training) or remote (async with polling)
        is_local = "localhost" in api_url or "127.0.0.1" in api_url
        
        if is_local:
            # No timeout for local synchronous training - it completes before returning
            timeout = None
        else:
            # Long timeout for remote async requests (if needed)
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
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get model details by ID."""
        return self._request("GET", f"/api/v1/models/{model_id}")
    
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
        max_epochs: int = 50,
        batch_size: int = 32,
        check_existing: bool = True,
    ) -> Dict[str, Any]:
        """Submit a training job."""
        # Check if model is already trained (if check_existing enabled)
        if check_existing:
            try:
                model = self.get_model(model_id)
                status = model.get("status", "unknown")
                if status == "trained":
                    return {
                        "status": "trained",
                        "message": "Model already trained",
                        "model_id": model_id,
                    }
            except Exception:
                # If we can't check, proceed anyway
                pass
        
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
        include_paths: bool = True,
    ) -> Dict[str, Any]:
        """Run a forecast."""
        payload = {
            "model_id": model_id,
            "scenario_name": scenario_name,
            "horizon": horizon,
            "n_paths": n_paths,
            "include_paths": include_paths,
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
        print_step("fetch", f"Conditioning: {result.get('features_processed', 0)} features, {result.get('total_data_points', 0)} points")
        
        # Fetch target data
        print_step("fetch", "Fetching target data...")
        result = self.client.fetch_data(
            self.target_set["id"],
            self.training_start,
            self.training_end,
        )
        print_step("fetch", f"Target: {result.get('features_processed', 0)} features, {result.get('total_data_points', 0)} points")
        
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
            check_existing=False,  # Don't skip - user explicitly requested training
        )
        
        status = result.get("status")
        message = result.get("message", "")
        print_step("train", f"Job submitted: status={status}")
        
        # Check for immediate failure (synchronous local training that failed)
        if status == "failed":
            print_error(f"Training failed immediately: {message}")
            return "failed"
        
        # If training runs synchronously (local dev), it's already done
        if status == "trained":
            print_success("Training completed (synchronous)")
            return "trained"
        
        # Poll for async training (status should be 'training' or 'pending')
        if status not in ["training", "pending"]:
            print_warning(f"Unexpected status after submission: {status}, will poll anyway")
        
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
            print_step("forecast", "Summary statistics (final timestep):")
            for stat in summary[:3]:  # First 3 features
                # Statistics are now time-series (list), get final value
                mean_final = stat['mean'][-1] if isinstance(stat['mean'], list) else stat['mean']
                std_final = stat['std'][-1] if isinstance(stat['std'], list) else stat['std']
                print(f"    {stat['feature']}: mean={mean_final:.4f}, std={std_final:.4f}")
        
        print_success("Unconditional forecast completed")
        
        # Plot forecast (always try to plot if we have summary data)
        if summary:
            # Try to get target feature from summary (if target_set not loaded)
            if self.target_set:
                target_features = [f['name'] for f in self.target_set.get('features', [])]
            else:
                # Use first feature from summary if target_set not available
                target_features = [summary[0]['feature']] if summary else []
            
            if target_features:
                plot_path = Path(__file__).parent / "forecast_unconditional.png"
                try:
                    _plot_forecast_paths(result, target_features[0], plot_path, scenario_name="Unconditional")
                    print_step("plot", f"Saved forecast plot to {plot_path}")
                except Exception as e:
                    print_warning(f"Failed to generate plot: {e}")
            else:
                print_warning("No target features available for plotting")
        else:
            print_warning("No summary data available for plotting")
        
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
        
        # Plot scenario forecast
        if self.target_set and self.target_set.get('features'):
            target_features = [f['name'] for f in self.target_set['features']]
            if target_features:
                plot_path = Path(__file__).parent / f"forecast_{scenario_name.lower().replace(' ', '_')}.png"
                try:
                    _plot_forecast_paths(result, target_features[0], plot_path, scenario_name=scenario_name)
                    print_step("plot", f"Saved scenario plot to {plot_path}")
                except Exception as e:
                    print_warning(f"Failed to generate scenario plot: {e}")
        
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
        choices=["all", "setup", "data_fetch", "model_creation", "training", "forecast"],
        default="all",
        help="Which stage to START from (previous stages skipped if data exists)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-detect and skip completed stages",
    )
    parser.add_argument("--model-id", help="Use existing model ID")
    parser.add_argument("--project-id", help="Use existing project ID")
    parser.add_argument("--conditioning-set-id", help="Use existing conditioning set ID")
    parser.add_argument("--target-set-id", help="Use existing target set ID")
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
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum training epochs (default: 50)")
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
    print(f"Stage: {args.stage}" + (" (resume mode)" if args.resume else ""))
    print(f"State Model: {args.state_model}")
    print(f"Generation Model: {args.generation_model}")
    
    client = SSMPipelineClient(args.api_url, args.api_key)
    
    # Define stage order and dependencies
    STAGES = ["setup", "data_fetch", "model_creation", "training", "forecast"]
    
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
        
        # Auto-discover existing resources if not provided
        # This makes resumption much easier - just run with --stage training
        
        # 1. Look up project by name
        if not args.project_id:
            try:
                projects = client._request("GET", "/api/v1/projects")
                for p in projects.get("projects", []):
                    if p.get("name") == args.project_name:
                        args.project_id = p["id"]
                        print(f"  🔍 Found existing project: {args.project_id}")
                        break
            except Exception:
                pass
        
        # 2. Look up feature sets from project
        if args.project_id and (not args.conditioning_set_id or not args.target_set_id):
            try:
                feature_sets = client._request("GET", f"/api/v1/feature-sets?project_id={args.project_id}")
                for fs in feature_sets.get("feature_sets", []):
                    if fs.get("set_type") == "conditioning" and not args.conditioning_set_id:
                        args.conditioning_set_id = fs["id"]
                        print(f"  🔍 Found existing conditioning set: {args.conditioning_set_id}")
                    elif fs.get("set_type") == "target" and not args.target_set_id:
                        args.target_set_id = fs["id"]
                        print(f"  🔍 Found existing target set: {args.target_set_id}")
            except Exception:
                pass
        
        # 3. Look up most recent model from project
        if args.project_id and not args.model_id:
            try:
                models = client._request("GET", f"/api/v1/models?project_id={args.project_id}")
                model_list = models.get("models", [])
                if model_list:
                    # Models are sorted by created_at DESC, so first is most recent
                    args.model_id = model_list[0]["id"]
                    print(f"  🔍 Found existing model (most recent): {args.model_id}")
            except Exception:
                pass
        
        # Apply discovered/provided IDs to test instance
        if args.project_id:
            test.project = {"id": args.project_id}
        if args.conditioning_set_id:
            test.conditioning_set = {"id": args.conditioning_set_id}
        if args.target_set_id:
            test.target_set = {"id": args.target_set_id}
        if args.model_id:
            test.model_id = args.model_id
            # Ensure model dict is also set (even if minimal)
            if not test.model:
                test.model = {"id": args.model_id}
        
        # Load full feature set details if IDs are known (needed for plotting)
        if args.target_set_id and (not test.target_set or 'features' not in test.target_set):
            try:
                test.target_set = client._request("GET", f"/api/v1/feature-sets/{args.target_set_id}")
            except Exception:
                pass
        if args.conditioning_set_id and (not test.conditioning_set or 'features' not in test.conditioning_set):
            try:
                test.conditioning_set = client._request("GET", f"/api/v1/feature-sets/{args.conditioning_set_id}")
            except Exception:
                pass
        
        if args.stage == "all":
            # Full pipeline
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
        
        else:
            # Run from specific stage
            start_idx = STAGES.index(args.stage)
            
            for stage in STAGES[start_idx:]:
                print_section(f"Running: {stage}")
                
                if stage == "setup":
                    if test.project:
                        print(f"  ⏭️  Skipping setup (project exists: {test.project['id']})")
                    else:
                        test.setup_infrastructure()
                
                elif stage == "data_fetch":
                    # Check if we need to fetch data
                    if test.conditioning_set and test.target_set:
                        # Check if data already exists
                        cs = client._request("GET", f"/api/v1/feature-sets/{test.conditioning_set['id']}")
                        ts = client._request("GET", f"/api/v1/feature-sets/{test.target_set['id']}")
                        if cs.get("fetched_data_available") and ts.get("fetched_data_available"):
                            print("  ⏭️  Skipping data fetch (data already available)")
                        else:
                            test.fetch_data()
                    else:
                        # Need setup first
                        if not test.project:
                            test.setup_infrastructure()
                        test.fetch_data()
                
                elif stage == "model_creation":
                    if test.model:
                        print(f"  ⏭️  Skipping model creation (model exists: {test.model['id']})")
                    else:
                        if not test.project or not test.conditioning_set or not test.target_set:
                            print_error("Need project and feature sets for model creation")
                            print("  Run with --stage setup or provide --project-id, --conditioning-set-id, --target-set-id")
                            sys.exit(1)
                        test.create_model(args.model_name)
                
                elif stage == "training":
                    # Check both model dict and model_id
                    if not test.model_id:
                        if test.model and test.model.get('id'):
                            test.model_id = test.model['id']
                        else:
                            print_error("Need model for training. Provide --model-id or run earlier stages.")
                            print("  Tip: Use --project-id to auto-discover the most recent model")
                            sys.exit(1)
                    
                    # Ensure model dict is set if we only have model_id
                    if not test.model or not test.model.get('id'):
                        try:
                            test.model = test.client.get_model(test.model_id)
                            print_step("model", f"Fetched model details: {test.model.get('name', test.model_id)}")
                        except Exception as e:
                            print_error(f"Could not fetch model {test.model_id}: {e}")
                            print("  Model may not exist or you may not have access")
                            sys.exit(1)
                    
                    # train_model will check if already trained and skip if so
                    status = test.train_model(
                        state_model=args.state_model,
                        generation_model=args.generation_model,
                        max_epochs=args.max_epochs,
                        poll_interval=args.poll_interval,
                        timeout=args.training_timeout,
                    )
                    
                    if status != "trained":
                        print_error(f"Training failed with status: {status}")
                        sys.exit(1)
                
                elif stage == "forecast":
                    if not test.model_id:
                        print_error("Need trained model for forecast. Provide --model-id or run earlier stages.")
                        sys.exit(1)
                    
                    # Check if model is trained before forecasting (when starting from this stage)
                    try:
                        model = test.client.get_model(test.model_id)
                        if model.get("status") != "trained":
                            print_error(f"Model {test.model_id} is not trained (status: {model.get('status')})")
                            print("  Run with --stage training to train the model first")
                            sys.exit(1)
                        print_success(f"Model {test.model_id} is trained - proceeding to forecast")
                    except Exception as e:
                        print_warning(f"Could not verify model status: {e}")
                        print("  Proceeding with forecast anyway...")
                    
                    test.run_forecast_unconditional(
                        horizon=args.forecast_horizon,
                        n_paths=args.n_paths,
                    )
                    
                    # Also run scenario forecast when starting from forecast stage
                    # (to test conditional forecasting with historical scenarios)
                    try:
                        test.run_forecast_with_scenario(
                            "Test Historical Scenario",
                            "VIX Volatility Index",
                            "2020-03-01",
                            "2020-03-21",
                            horizon=args.forecast_horizon,
                            n_paths=args.n_paths,
                        )
                        print_success("Scenario forecast completed")
                    except Exception as e:
                        print_warning(f"Scenario forecast failed (optional): {e}")
                
                # Print IDs after each stage for easy resumption
                print("\n  📋 Current state (use these to resume):")
                if test.project:
                    print(f"     --project-id {test.project['id']}")
                if test.conditioning_set:
                    print(f"     --conditioning-set-id {test.conditioning_set['id']}")
                if test.target_set:
                    print(f"     --target-set-id {test.target_set['id']}")
                if test.model:
                    print(f"     --model-id {test.model['id']}")
    
    finally:
        client.close()


if __name__ == "__main__":
    main()

