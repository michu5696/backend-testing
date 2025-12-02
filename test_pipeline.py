#!/usr/bin/env python3
"""
End-to-end helper to build and test a Treasury ETF model directly via the backend HTTP APIs.

This script performs the following steps:
  1. Creates a project with the desired training window.
  2. Creates conditioning/target feature sets and registers FRED/Yahoo data collectors.
  3. Adds the Treasury ETF + macro features used by the template model.
  4. Fetches historical data for both feature sets.
  5. Creates a model, generates samples and trains a vine-copula model via /api/v1/ml/train.
  6. Creates a "covid" scenario (simulation date 2020-03-15) and runs a forecast with
     past windows in RECENT mode and future conditioning windows anchored to the scenario date.
  7. Persists a summary JSON file under backend-testing/artifacts/ so it can be reused.

Requirements:
  * A running backend (e.g., `uvicorn main:app --reload` from backend/).
  * A valid admin API key with access to the new backend (defaults to BACKEND_ADMIN_API_KEY
    or backend/admin/admin_api_key_v2.txt).
  * Network access to FRED/Yahoo for data ingestion.

Example:
    python treasury_workflow.py \
        --api-url http://localhost:8000 \
        --api-key "$(cat ../backend/admin/admin_api_key_v2.txt)" \
        --project-name "Treasury ETFs (API)" \
        --simulation-date 2020-03-15
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
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

DEFAULT_CONDITIONING_FEATURES: List[Dict[str, str]] = [
    {"id": "^IRX", "source": "yahoo", "name": "13-Week Treasury Yield"},
    {"id": "^FVX", "source": "yahoo", "name": "5-Year Treasury Yield"},
    {"id": "^TNX", "source": "yahoo", "name": "10-Year Treasury Yield"},
    {"id": "T10Y2Y", "source": "fred", "name": "10-Year Minus 2-Year Treasury Spread"},
    {"id": "T10YIE", "source": "fred", "name": "10-Year Breakeven Inflation Rate"},
    {"id": "BAMLC0A0CM", "source": "fred", "name": "Investment Grade Corporate Spread"},
    {"id": "VIXCLS", "source": "fred", "name": "VIX Volatility Index"},
    {"id": "UNRATE", "source": "fred", "name": "Unemployment Rate"},
]

DEFAULT_TARGET_FEATURES: List[Dict[str, str]] = [
    {"id": "SHY", "source": "yahoo", "name": "1-3 Year Treasury Bond ETF"},
    {"id": "IEI", "source": "yahoo", "name": "3-7 Year Treasury Bond ETF"},
    {"id": "IEF", "source": "yahoo", "name": "7-10 Year Treasury Bond ETF"},
    {"id": "TLT", "source": "yahoo", "name": "20+ Year Treasury Bond ETF"},
]

def load_default_api_key() -> Optional[str]:
    base_dir = Path(__file__).resolve().parents[1] / "backend" / "admin"
    v2_candidate = base_dir / "admin_api_key_v2.txt"
    legacy_candidate = base_dir / "admin_api_key.txt"
    if v2_candidate.exists():
        return v2_candidate.read_text().strip()
    if legacy_candidate.exists():
        return legacy_candidate.read_text().strip()
    return None


class TreasuryWorkflow:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        project_name: str,
        description: str,
        training_start: str,
        training_end: str,
        conditioning_features: List[Dict[str, str]],
        target_features: List[Dict[str, str]],
        fred_api_key: str,
        sample_config: Dict[str, Any],
        scenario_date: str,
        scenario_name: str,
        model_name: str,
        model_description: str,
        project_id_override: Optional[str] = None,
        model_id_override: Optional[str] = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.project_name = project_name
        self.description = description
        self.training_start = training_start
        self.training_end = training_end
        self.conditioning_features = conditioning_features
        self.target_features = target_features
        self.fred_api_key = fred_api_key
        self.sample_config = sample_config
        self.scenario_date = scenario_date
        self.scenario_name = scenario_name
        self.model_name = model_name
        self.model_description = model_description
        # No timeout - wait as long as needed
        timeout = httpx.Timeout(None, connect=30.0)
        self.session = httpx.Client(
            base_url=self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

        self.user_id: Optional[str] = None
        self.project_id_override = project_id_override
        self.model_id: Optional[str] = model_id_override
        self.project: Optional[Dict[str, Any]] = {"id": project_id_override} if project_id_override else None
        self.scenario: Optional[Dict[str, Any]] = None

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        # Ensure every call has a timeout so we don't hang forever waiting for a response
        if "timeout" not in kwargs:
            kwargs["timeout"] = 30.0

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
            print(f"[api] ⚠️  Timeout requesting {path}")
            raise

        if resp.content:
            return resp.json()
        return {}

    def resolve_user(self) -> None:
        data = self._request("GET", "/api/v1/account/user")
        self.user_id = data["user_id"]
        print(f"[account] user: {self.user_id} ({data.get('email', 'unknown email')})")

    def create_project(self) -> None:
        if self.project_id_override:
            self.project = self.project or {"id": self.project_id_override}
            print(f"[project] using existing project {self.project['id']}")
            return
        
        # Try to find existing project by name
        try:
            projects_response = self._request("GET", "/api/v1/projects/?limit=100&offset=0")
            existing_projects = projects_response.get("projects", [])
            for proj in existing_projects:
                if proj.get("name") == self.project_name:
                    self.project = proj
                    print(f"[project] reusing existing project {self.project['id']} ({self.project['name']})")
                    
                    # Update date range if different
                    current_start = proj.get("training_start_date")
                    current_end = proj.get("training_end_date")
                    if current_start != self.training_start or current_end != self.training_end:
                        print(f"[project] updating date range: {current_start} to {current_end} → {self.training_start} to {self.training_end}")
                        update_payload = {
                            "training_start_date": self.training_start,
                            "training_end_date": self.training_end,
                        }
                        self.project = self._request("PATCH", f"/api/v1/projects/{self.project['id']}", json=update_payload)
                        print(f"[project] date range updated")
                    else:
                        print(f"[project] date range unchanged: {current_start} to {current_end}")
                    
                    return
        except Exception as e:
            print(f"[project] could not check for existing projects: {e}")
        
        # Create new project if not found
        payload = {
            "name": self.project_name,
            "description": self.description,
            "training_start_date": self.training_start,
            "training_end_date": self.training_end,
            "is_template": True,
        }
        self.project = self._request("POST", "/api/v1/projects/", json=payload)
        print(f"[project] created {self.project['id']} ({self.project['name']})")

    def _build_auto_setup_payload(self) -> Dict[str, Any]:
        conditioning_collectors = [
            {"source": "fred", "api_key": self.fred_api_key},
            {"source": "yahoo"},
        ]
        target_collectors = [{"source": "yahoo"}]

        return {
            "project_id": self.project["id"],
            "conditioning_set": {
                "name": "US Macro Indicators",
                "description": "Macro + market indicators for Treasury ETF forecasting",
                "data_collectors": conditioning_collectors,
                "features": self.conditioning_features,
            },
            "target_set": {
                "name": "US Treasury ETFs",
                "description": "Purchasable Treasury ETFs across the curve",
                "data_collectors": target_collectors,
                "features": self.target_features,
            },
            "model": {
                "name": self.model_name,
                "description": self.model_description,
            },
        }

    def ensure_model_exists(self) -> None:
        """Ensure model exists (create if needed using auto_setup, or reuse existing)."""
        if self.model_id:
            return  # Already have model_id
        
        if not self.project:
            raise RuntimeError("Project must be created before ensuring model exists")
        
        # Try to find existing model by name in the project
        try:
            models_resp = self._request("GET", f"/api/v1/models/?project_id={self.project['id']}&limit=100")
            models = models_resp.get("models", [])
            for model in models:
                if model.get("name") == self.model_name:
                    self.model_id = model["id"]
                    print(f"[model] reusing existing model {self.model_id} ({self.model_name})")
                    return
        except Exception as e:
            print(f"[model] could not check for existing models: {e}")
        
        # Model doesn't exist, create it using auto_setup (but do it synchronously here)
        print(f"[model] creating model {self.model_name}...")
        auto_setup = self._build_auto_setup_payload()
        
        # Create feature sets and model via a separate endpoint or by calling the backend directly
        # For now, we'll use auto_setup in the training call but it should be fast if resources exist
        # Actually, let's create them explicitly via API if endpoints exist
        # If not, we'll let auto_setup handle it (but it should be fast)
        print(f"[model] model will be created via auto_setup during training")
        # Model will be created in Cloud Task, but we need model_id for polling
        # So we'll use auto_setup and handle "pending" model_id

    def train_with_auto_setup(
        self,
        n_regimes: int,
        poll_interval: int,
        timeout: int,
        resume_from: str = None,
    ) -> None:
        """
        Train the model. If model_id is set, uses it directly. Otherwise uses auto_setup.
        
        Args:
            n_regimes: Number of copula regimes
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            resume_from: Optional stage to resume from ('sample_generation', 'fitting', 'encoding', 'training')
                        None = full pipeline (default)
                        'sample_generation' = regenerate normalized samples
                        'fitting' = refit encoding models (PCA-ICA)
                        'encoding' = re-encode samples with existing models
                        'training' = only retrain copula (fastest)
        
        Note: ICA (temporal decorrelation) is always applied.
              PCA (feature compression) is automatically applied only to multivariate groups.
        """
        if not self.project:
            raise RuntimeError("Project must be created or specified before training")
        
        # Build payload - use model_id if available, otherwise auto_setup
        train_payload = {
            "n_regimes": n_regimes,
            "data_fetch": {
                "start_date": self.training_start,
                "end_date": self.training_end,
                "api_keys": {"fred": self.fred_api_key},
            },
            "sample": {
                "pastWindow": self.sample_config["pastWindow"],
                "futureWindow": self.sample_config["futureWindow"],
                "stride": self.sample_config["stride"],
                "splitPercentages": self.sample_config["splitPercentages"],
            },
        }
        
        # Ensure model exists first (reuse existing or will be created via auto_setup)
        self.ensure_model_exists()
        
        # Use model_id if we have it (faster - no auto_setup needed)
        if self.model_id:
            train_payload["model_id"] = self.model_id
            print(f"[train] Using existing model_id: {self.model_id}")
        else:
            # Model doesn't exist yet - use auto_setup (will be created in Cloud Task)
            train_payload["auto_setup"] = self._build_auto_setup_payload()
            print(f"[train] Using auto_setup (model will be created in Cloud Task)")
        
        # Add resume_from if specified
        if resume_from:
            train_payload["resume_from"] = resume_from
            print(f"[train] Resuming pipeline from stage: {resume_from}")
        
        print(f"[train] Submitting training job to {self.api_url}...")
        print(f"[train] Payload: n_regimes={n_regimes}, stride={self.sample_config['stride']}, date_range={self.training_start} to {self.training_end}")
        if "auto_setup" in train_payload:
            print(f"[train] Note: Model will be created via auto_setup in Cloud Task (non-blocking)")
        else:
            print(f"[train] Note: Using existing model, submission should return immediately")
        
        # Submit job - use a reasonable timeout for submission
        # Auto-setup operations (creating feature sets/models) can be slow, especially under load
        import time as time_module
        submission_start = time_module.time()
        last_log_time = submission_start
        
        def log_progress():
            elapsed = time_module.time() - submission_start
            if elapsed > 30:  # Log every 30 seconds if taking a while
                print(f"[train] Still waiting for submission response... ({elapsed:.0f}s elapsed)")
        
        try:
            # Use a longer timeout for submission (10 minutes) to allow for auto-setup operations
            # The backend needs to create feature sets, models, etc. which can take time under load
            submit_timeout = httpx.Timeout(600.0, connect=60.0)  # 10 min total, 1 min connect
            
            # Start a background thread to log progress (will be stopped after response)
            import threading
            progress_stop = threading.Event()
            progress_thread = None
            
            def progress_logger():
                while not progress_stop.is_set():
                    if progress_stop.wait(timeout=30):  # Wait 30s or until stopped
                        break
                    elapsed = time_module.time() - submission_start
                    if elapsed < 600:  # Only log if still within timeout
                        print(f"[train] ⏳ Still waiting for submission response... ({elapsed:.0f}s elapsed)")
            
            progress_thread = threading.Thread(target=progress_logger, daemon=True)
            progress_thread.start()
            
            try:
                with httpx.Client(
                    base_url=self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=submit_timeout,
                ) as submit_session:
                    print(f"[train] Sending POST request (timeout: 10min for auto-setup operations)...")
                    resp = submit_session.post("/api/v1/ml/train", json=train_payload)
                    submission_elapsed = time_module.time() - submission_start
                    print(f"[train] ✅ Received response after {submission_elapsed:.1f}s")
                    
                    # Stop progress logger now that we have response
                    progress_stop.set()
                    
                    if resp.status_code >= 400:
                        try:
                            detail = resp.json()
                        except Exception:
                            detail = resp.text
                        raise RuntimeError(f"POST /api/v1/ml/train failed ({resp.status_code}): {detail}")
                    job = resp.json() if resp.content else {}
            finally:
                # Ensure progress logger is stopped
                progress_stop.set()
        except httpx.TimeoutException:
            submission_elapsed = time_module.time() - submission_start
            raise RuntimeError(
                f"Training job submission timed out after {submission_elapsed:.1f}s. "
                f"The backend may be slow during auto-setup (creating feature sets/models) or blocked by other operations. "
                f"Check backend logs. The job may still be queued in Cloud Tasks even if this times out."
            )
        except Exception as e:
            submission_elapsed = time_module.time() - submission_start
            raise RuntimeError(f"Failed to submit training job after {submission_elapsed:.1f}s: {str(e)}")
        
        self.model_id = job.get("model_id")
        job_id = job.get("job_id")
        initial_status = job.get("status", "unknown")
        
        if not job_id:
            raise RuntimeError(f"Job submission failed: no job_id in response. Response: {job}")
        
        # If model_id is "pending", it means auto_setup is happening in the Cloud Task
        # Wait a bit for it to resolve, then get the real model_id from status
        if self.model_id == "pending" or not self.model_id:
            print(f"[train] ⏳ Model ID is pending (auto-setup in progress), waiting for resolution...")
            import time as time_module
            for attempt in range(10):  # Wait up to 10 seconds
                time_module.sleep(1)
                # Try to get status - but we need model_id... 
                # Actually, we can't poll without model_id. Let's wait and hope it resolves quickly.
                # Or we could list models and find the one with matching job_id in metadata
                # For now, just wait a bit more
                if attempt == 9:
                    print(f"[train] ⚠️  Model ID still pending after 10s, will try to resolve from project...")
                    # Try to find model by listing models in the project
                    if self.project:
                        models_resp = self._request("GET", f"/api/v1/models/?project_id={self.project['id']}")
                        models = models_resp.get("models", [])
                        # Find the most recently created one (should be ours)
                        if models:
                            # Sort by created_at if available, or just take the first
                            self.model_id = models[0]["id"]
                            print(f"[train] ✅ Resolved model ID: {self.model_id}")
                        else:
                            raise RuntimeError("Could not resolve model_id after auto-setup. Check backend logs.")
        
        if not self.model_id:
            raise RuntimeError(f"Job submission failed: no model_id in response. Response: {job}")
        
        print(f"[train] ✅ Job submitted successfully!")
        print(f"[train]   Job ID: {job_id}")
        print(f"[train]   Model ID: {self.model_id}")
        print(f"[train]   Initial status: {initial_status}")
        print(f"[train]   Starting polling (interval={poll_interval}s, timeout={'unlimited' if timeout <= 0 else f'{timeout}s'})...")
        print()
        
        self._poll_training_status(poll_interval, timeout, expected_job_id=job_id)

    def ensure_model_for_forecast(self) -> None:
        if self.model_id:
            return

        response = self._request("GET", "/api/v1/models/?limit=100&offset=0")
        models = response.get("models", [])
        if not models:
            raise RuntimeError("No trained models available. Run the training stage first or provide --model-id.")

        selected = None
        project_id = self.project.get("id") if self.project else None
        if project_id:
            for candidate in models:
                if candidate.get("project_id") == project_id:
                    selected = candidate
                    break
        if not selected:
            selected = models[0]

        self.model_id = selected["id"]
        if not self.project:
            pid = selected.get("project_id")
            if pid:
                self.project = {"id": pid}
        print(f"[model] using existing model {self.model_id}")

    def _poll_training_status(self, interval_seconds: int, timeout_seconds: int, expected_job_id: Optional[str] = None) -> None:
        if not self.model_id:
            raise RuntimeError("Model ID is not set; cannot poll status")

        elapsed = 0
        status = "unknown"
        last_status = None
        poll_count = 0
        job_id_mismatch_count = 0
        
        # If timeout is None or <= 0, poll indefinitely
        while timeout_seconds is None or timeout_seconds <= 0 or elapsed <= timeout_seconds:
            try:
                resp = self._request(
                    "GET",
                    f"/api/v1/ml/train/{self.model_id}/status",
                )
                status = resp.get("status", "unknown")
                current_job_id = resp.get("job_id")
                message = resp.get("message")
                
                # Check if we're tracking the right job (if expected_job_id is provided)
                if expected_job_id and current_job_id and current_job_id != expected_job_id:
                    job_id_mismatch_count += 1
                    if job_id_mismatch_count == 1:
                        print(f"[train] ⚠️  WARNING: Status endpoint returned different job_id ({current_job_id}) than expected ({expected_job_id})")
                        print(f"[train] ⚠️  This may indicate an old job is still running. Continuing to poll...")
                    # Continue polling - the job might still be queued
                
                # Always print status for visibility
                status_line = f"[train] status={status} (t={elapsed}s, poll #{poll_count + 1})"
                if current_job_id:
                    status_line += f" [job: {current_job_id[:16]}...]"
                if message:
                    status_line += f" - {message}"
                print(status_line)
                last_status = status
                
                poll_count += 1
                
                # Only break if status is terminal AND we're tracking the right job (or job_id check is disabled)
                if status in {"completed", "failed", "training-failed"}:
                    if not expected_job_id or current_job_id == expected_job_id:
                        break
                    # If wrong job completed, continue polling
                    
            except Exception as e:
                print(f"[train] ⚠️  Error polling status (t={elapsed}s): {str(e)}")
                # Continue polling even if one request fails
                
            time.sleep(interval_seconds)
            elapsed += interval_seconds
        
        print()
        if status == "completed":
            print(f"[train] ✅ Training completed successfully! (total time: {elapsed}s)")
        elif status in {"failed", "training-failed"}:
            raise RuntimeError(f"Training failed with status: {status}")
        else:
            raise RuntimeError(f"Training did not complete (status={status}, elapsed={elapsed}s)")

    def create_scenario(self) -> None:
        if not self.model_id:
            raise RuntimeError("Model ID is not set; cannot create scenario")
        payload = {
            "model_id": self.model_id,
            "name": self.scenario_name,
            "description": "COVID stress scenario anchored to March 15 2020",
            "simulation_date": self.scenario_date,
        }
        self.scenario = self._request("POST", "/api/v1/scenarios/", json=payload)
        print(f"[scenario] created {self.scenario['id']} ({self.scenario['name']})")

    def run_forecast(self, n_paths: int) -> None:
        if not self.model_id or not self.scenario:
            raise RuntimeError("Scenario and model must be available before forecasting")

        conditioning_names = [feature["name"] for feature in self.conditioning_features]
        target_names = [feature["name"] for feature in self.target_features]

        directives = []
        for feature in conditioning_names:
            directives.append(
                {
                    "feature": feature,
                    "window": "future_conditioning",
                    "mode": "simulation_date",
                    "simulation_date": self.scenario_date,
                }
            )
        for feature in target_names:
            directives.append(
                {"feature": feature, "window": "future_target", "mode": "unobserved"}
            )

        forecast_payload = {
            "user_id": self.user_id,
            "model_id": self.model_id,
            "scenario_id": self.scenario["id"],
            "default_past_mode": "recent",
            "default_future_mode": "simulation_date",
            "default_future_simulation_date": self.scenario_date,
            "conditioning": directives,
            "n_paths": n_paths,
            "random_seed": 42,
        }

        # Forecast can take a long time, especially with adaptation logic
        forecast = self._request(
            "POST",
            "/api/v1/ml/forecast",
            json=forecast_payload,
            timeout=None,  # No timeout - wait as long as needed
        )
        print(
            f"[forecast] status={forecast.get('status')} "
            f"paths={len(forecast.get('paths', []))}"
        )
        plot_path = Path(__file__).parent / "forecast_preview.png"
        primary_conditioning = self.conditioning_features[0]["name"]
        primary_target = self.target_features[0]["name"]
        _maybe_plot_forecast(forecast, primary_conditioning, primary_target, plot_path)

    def close(self) -> None:
        self.session.close()


def _maybe_plot_forecast(
    forecast_data: Dict[str, Any],
    conditioning_feature: str,
    target_feature: str,
    output_path: Path,
) -> None:
    """Plot historical + forecast trajectories, mirroring forecast_smoke style."""
    if plt is None:
        print("[plot] matplotlib not available; skipping forecast plot")
        return

    reconstructed = forecast_data.get("reconstructed_paths") or []
    historical = (forecast_data.get("reference_context") or {}).get("historical_reconstructed") or []

    if not reconstructed and not historical:
        print("[plot] no reconstructed data returned; skipping")
        return

    plot_feature = target_feature
    past_series = None
    for entry in historical:
        if entry.get("feature") == plot_feature and entry.get("temporal_tag") == "past":
            values = entry.get("reconstructed_values") or []
            if values:
                past_series = list(values)
                break

    if past_series is None:
        for entry in historical:
            if entry.get("feature") == conditioning_feature and entry.get("temporal_tag") == "past":
                values = entry.get("reconstructed_values") or []
                if values:
                    past_series = list(values)
                    plot_feature = conditioning_feature
                    break

    future_paths: Dict[int, List[float]] = {}
    for entry in reconstructed:
        if entry.get("feature") == target_feature and entry.get("temporal_tag") == "future":
            idx = entry.get("_sample_idx")
            values = entry.get("reconstructed_values") or []
            if idx is not None and values:
                future_paths[idx] = list(values)

    future_dates = forecast_data.get("future_dates") or []
    past_dates = forecast_data.get("past_dates") or []

    if not past_series:
        print("[plot] insufficient past data to plot")
        return
    if not future_paths or not future_dates:
        print("[plot] insufficient future data to plot")
        return

    past_dates = past_dates[-len(past_series):] if past_dates else []
    first_future = next(iter(future_paths.values()))
    horizon = len(first_future)
    future_dates = future_dates[:horizon]

    sorted_indices = sorted(future_paths.keys())
    forecasts_array = np.array([future_paths[idx] for idx in sorted_indices])
    n_samples, n_timesteps = forecasts_array.shape

    def _parse_date(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.strptime(value, "%Y-%m-%d")

    if past_dates and future_dates:
        past_t = [_parse_date(d) for d in past_dates]
        future_t = [_parse_date(d) for d in future_dates]
        use_dates = True
        all_dates = past_t + future_t
    else:
        past_t = np.arange(len(past_series))
        future_t = np.arange(len(past_series), len(past_series) + n_timesteps)
        use_dates = False
        all_dates = list(past_t) + list(future_t)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(past_t, past_series, "o-", color="black", linewidth=2, markersize=4, alpha=0.8, label="Recent Past", zorder=5)
    if not use_dates:
        ax.axvline(x=len(past_series), color="red", linestyle=":", linewidth=2, alpha=0.5, label="Forecast Start", zorder=4)

    # Plot historical target future path (from simulation date) if available
    historical_future_path = None
    for entry in historical:
        if (entry.get("feature") == target_feature and 
            entry.get("temporal_tag") == "future" and 
            entry.get("_is_historical_pattern") == True):
            values = entry.get("reconstructed_values") or []
            if values:
                historical_future_path = list(values)
                break
    
    if historical_future_path:
        # Get simulation date for label
        conditioning_summary = forecast_data.get("conditioning_summary", {})
        simulation_date = conditioning_summary.get("default_future_simulation_date")
        hist_label = f"Historical Path ({simulation_date})" if simulation_date else "Historical Path"
        
        # Plot with SDK style: green, o-, linewidth=2.5, markersize=5, alpha=0.9, zorder=6
        ax.plot(future_t[:len(historical_future_path)], historical_future_path, "o-", 
               color="green", linewidth=2.5, markersize=5, alpha=0.9, 
               label=hist_label, zorder=6)

    n_to_plot = min(50, n_samples)
    for i in range(n_to_plot):
        ax.plot(future_t, forecasts_array[i], "-", alpha=0.2, linewidth=0.8, color="steelblue", zorder=2)
    ax.plot([], [], "-", alpha=0.5, linewidth=1.5, color="steelblue", label=f"Forecast Paths (n={n_samples})", zorder=2)

    for ci_level, color, alpha in [(0.68, "darkblue", 0.2), (0.95, "steelblue", 0.15)]:
        lower_q = (1 - ci_level) / 2
        upper_q = 1 - lower_q
        lower = np.percentile(forecasts_array, lower_q * 100, axis=0)
        upper = np.percentile(forecasts_array, upper_q * 100, axis=0)
        ax.fill_between(future_t, lower, upper, alpha=alpha, color=color, label=f"{int(ci_level*100)}% CI", zorder=3)

    median_forecast = np.median(forecasts_array, axis=0)
    ax.plot(future_t, median_forecast, "-", color="darkred", linewidth=2.5, alpha=0.9, label="Median Forecast", zorder=7)

    ax.set_title(f"{target_feature} - Conditional Forecast", fontsize=14, fontweight="bold")
    if use_dates:
        ax.set_xlabel("Date", fontsize=12)
        if mdates is not None:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.set_xlabel("Time Step", fontsize=12)

    ax.set_ylabel(f"{target_feature} Value", fontsize=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    stats_text = f"Min: {np.min(forecasts_array):.3f}\nMax: {np.max(forecasts_array):.3f}\nMedian: {np.median(median_forecast):.3f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved forecast preview to {output_path}")

    last_past = past_series[-1]
    first_future = forecasts_array[0, 0]
    print(f"[plot] continuity check: last past {last_past:.4f} → first future {first_future:.4f}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and test Treasury ETF models via backend APIs")
    parser.add_argument("--api-url", default=os.environ.get("SABLIER_API_URL", "https://sablier-api-v2-215397666394.us-central1.run.app"))
    parser.add_argument("--stage", choices=["all", "training", "forecast"], default="all")
    parser.add_argument("--project-id", help="Existing project to reuse (will be reused if found by name)")
    parser.add_argument("--model-id", help="Existing model to reuse (will be reused if found by name)")
    parser.add_argument("--reuse-project", action="store_true", default=True, help="Reuse existing project by name if it exists")
    parser.add_argument("--reuse-model", action="store_true", default=True, help="Reuse existing model by name if it exists")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SABLIER_API_KEY") or load_default_api_key(),
        help="Admin API key (default reads backend/admin/admin_api_key_v2.txt)",
    )
    parser.add_argument("--fred-api-key", default=os.environ.get("FRED_API_KEY", "3d349d18e62cab7c2d55f1a6680f06d8"))
    parser.add_argument("--project-name", default="main project")
    parser.add_argument("--project-description", default="Main project containing all available feature sets and models")
    parser.add_argument("--training-start", default="2015-01-01")
    parser.add_argument("--training-end", default="2025-09-01")
    parser.add_argument("--model-name", default="Treasury ETF Forecasting Model (API)")
    parser.add_argument(
        "--model-description",
        default="Treasury ETF forecasting model created via backend APIs",
    )
    parser.add_argument(
        "--new-model",
        action="store_true",
        default=False,
        help="Create a new model instead of reusing existing one with the same name. Appends timestamp to model name."
    )
    parser.add_argument("--scenario-name", default="covid")
    parser.add_argument("--scenario-date", default="2020-03-15")
    parser.add_argument("--past-window", type=int, default=100)
    parser.add_argument("--future-window", type=int, default=80)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--training-split", type=float, default=0.8, help="Fraction allocated to training split")
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--training-timeout", type=int, default=0, help="Seconds to wait for training completion (0 = no timeout)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling cadence for training status")
    parser.add_argument(
        "--resume-from",
        choices=["sample_generation", "fitting", "encoding", "training"],
        default=None,
        help="Resume training pipeline from a specific stage (skips earlier stages if data exists). "
             "'sample_generation' = regenerate samples, 'fitting' = refit encoders, "
             "'encoding' = re-encode samples, 'training' = only retrain copula"
    )
    parser.add_argument("--forecast-paths", type=int, default=100, help="Number of forecast trajectories")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.api_key:
        print("ERROR: API key not provided. Use --api-key or set SABLIER_API_KEY.", file=sys.stderr)
        sys.exit(1)

    training_split = max(0.1, min(args.training_split, 0.99))
    validation_split = round(1.0 - training_split, 4)

    sample_config = {
        "pastWindow": args.past_window,
        "futureWindow": args.future_window,
        "stride": args.stride,
        "splitPercentages": {"training": training_split, "validation": validation_split}
    }

    # If --new-model flag is set, append timestamp to model name to ensure uniqueness
    model_name = args.model_name
    if args.new_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.model_name} ({timestamp})"
        print(f"[model] Creating new model with name: {model_name}")

    workflow = TreasuryWorkflow(
        api_url=args.api_url,
        api_key=args.api_key,
        project_name=args.project_name,
        description=args.project_description,
        training_start=args.training_start,
        training_end=args.training_end,
        conditioning_features=DEFAULT_CONDITIONING_FEATURES,
        target_features=DEFAULT_TARGET_FEATURES,
        fred_api_key=args.fred_api_key,
        sample_config=sample_config,
        scenario_date=args.scenario_date,
        scenario_name=args.scenario_name,
        model_name=model_name,
        model_description=args.model_description,
        project_id_override=args.project_id,
        model_id_override=args.model_id,
    )

    workflow.resolve_user()

    try:
        if args.stage in {"all", "training"}:
            workflow.create_project()
            workflow.train_with_auto_setup(
                n_regimes=args.n_regimes,
                poll_interval=args.poll_interval,
                timeout=args.training_timeout,
                resume_from=args.resume_from,
            )

        if args.stage in {"all", "forecast"}:
            if args.stage == "forecast":
                workflow.ensure_model_for_forecast()
            workflow.create_scenario()
            workflow.run_forecast(n_paths=args.forecast_paths)
    finally:
        workflow.close()


if __name__ == "__main__":
    main()

