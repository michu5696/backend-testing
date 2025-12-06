#!/usr/bin/env python3
"""
Script to create models from combinations of conditioning and target sets
and submit training jobs for each model.

Creates 8 models:
- 2 diversified target sets × 4 conditioning sets each
- Conditioning sets: 2 non-diversified + 2 diversified
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import date
from typing import Dict, List, Any, Optional

# Add backend directory to path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_dir)

# Load environment variables from backend/.env
from dotenv import load_dotenv
env_path = Path(backend_dir) / '.env'
load_dotenv(dotenv_path=env_path)

from cloudsql_client import CloudSQLManager
import logging
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
# Read API key from admin file
admin_key_path = Path(backend_dir) / 'admin' / 'admin_api_key_v2.txt'
if admin_key_path.exists():
    API_KEY = admin_key_path.read_text().strip()
    logger.info(f"✅ Loaded API key from {admin_key_path}")
else:
    API_KEY = os.getenv("API_KEY", "")
    if not API_KEY:
        logger.warning("⚠️  API key file not found and API_KEY env var not set")

# Use cloud API URL - ensure HTTPS
api_url_raw = os.getenv("API_URL", "https://sablier-api-v2-z4kedk7vaa-uc.a.run.app")
# Force HTTPS for Cloud Run URLs (security)
if api_url_raw.startswith("http://") and "run.app" in api_url_raw:
    API_URL = api_url_raw.replace("http://", "https://")
    logger.warning(f"⚠️  Forced HTTPS for API URL: {api_url_raw} -> {API_URL}")
else:
    API_URL = api_url_raw
logger.info(f"Using API URL: {API_URL}")

# FRED API Key - check environment variable, .env file, or use default
FRED_API_KEY = os.getenv("FRED_API_KEY", "3d349d18e62cab7c2d55f1a6680f06d8")
if FRED_API_KEY:
    logger.info(f"✅ Using FRED API key (from env or default)")
else:
    logger.warning("⚠️  FRED_API_KEY not found. Training may fail if user doesn't have stored FRED key.")

# Model configurations
MODEL_CONFIGS = [
    # Target Set 1: Balanced Multi-Asset Portfolio
    {
        "target_set_name": "Balanced Multi-Asset Portfolio",
        "conditioning_set_name": "Core Macroeconomic Indicators",  # Non-diversified
        "model_name": "Balanced Portfolio - Core Macro Indicators",
        "model_description": "Balanced multi-asset portfolio model conditioned on core macroeconomic indicators (rates, inflation, GDP, unemployment)",
    },
    {
        "target_set_name": "Balanced Multi-Asset Portfolio",
        "conditioning_set_name": "Interest Rates & Yield Curve",  # Non-diversified
        "model_name": "Balanced Portfolio - Interest Rates & Yield Curve",
        "model_description": "Balanced multi-asset portfolio model conditioned on comprehensive interest rates and yield curve metrics",
    },
    {
        "target_set_name": "Balanced Multi-Asset Portfolio",
        "conditioning_set_name": "Core Macroeconomic Mix",  # Diversified
        "model_name": "Balanced Portfolio - Core Macro Mix",
        "model_description": "Balanced multi-asset portfolio model conditioned on diversified core macroeconomic mix (rates, inflation, GDP, unemployment, volatility)",
    },
    {
        "target_set_name": "Balanced Multi-Asset Portfolio",
        "conditioning_set_name": "Rates, Inflation & Currency Mix",  # Diversified
        "model_name": "Balanced Portfolio - Rates, Inflation & Currency Mix",
        "model_description": "Balanced multi-asset portfolio model conditioned on diversified rates, inflation, and currency factors",
    },
    # Target Set 2: Comprehensive Multi-Asset Portfolio
    {
        "target_set_name": "Comprehensive Multi-Asset Portfolio",
        "conditioning_set_name": "Core Macroeconomic Indicators",  # Non-diversified
        "model_name": "Comprehensive Portfolio - Core Macro Indicators",
        "model_description": "Comprehensive multi-asset portfolio model conditioned on core macroeconomic indicators (rates, inflation, GDP, unemployment)",
    },
    {
        "target_set_name": "Comprehensive Multi-Asset Portfolio",
        "conditioning_set_name": "Interest Rates & Yield Curve",  # Non-diversified
        "model_name": "Comprehensive Portfolio - Interest Rates & Yield Curve",
        "model_description": "Comprehensive multi-asset portfolio model conditioned on comprehensive interest rates and yield curve metrics",
    },
    {
        "target_set_name": "Comprehensive Multi-Asset Portfolio",
        "conditioning_set_name": "Core Macroeconomic Mix",  # Diversified
        "model_name": "Comprehensive Portfolio - Core Macro Mix",
        "model_description": "Comprehensive multi-asset portfolio model conditioned on diversified core macroeconomic mix (rates, inflation, GDP, unemployment, volatility)",
    },
    {
        "target_set_name": "Comprehensive Multi-Asset Portfolio",
        "conditioning_set_name": "Rates, Inflation & Currency Mix",  # Diversified
        "model_name": "Comprehensive Portfolio - Rates, Inflation & Currency Mix",
        "model_description": "Comprehensive multi-asset portfolio model conditioned on diversified rates, inflation, and currency factors",
    },
]


async def set_user_fred_api_key(
    api_url: str,
    api_key: str,
    fred_api_key: str,
) -> bool:
    """Set FRED API key for the authenticated user"""
    if not fred_api_key:
        logger.warning("⚠️  No FRED API key provided, skipping user API key setup")
        return False
    
    url = f"{api_url}/api/v1/user-api-keys"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    payload = {
        "provider": "fred",
        "api_key": fred_api_key,
        "name": "FRED API Key",
    }
    
    logger.info("Setting FRED API key for user...")
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            logger.info(f"✅ FRED API key set successfully (key ID: {result.get('id')})")
            return True
    except httpx.HTTPStatusError as e:
        # Check if it's a 400 (might mean already exists or validation error)
        if e.response.status_code == 400:
            logger.info("ℹ️  FRED API key may already be set or validation failed")
            return True
        logger.error(f"❌ Failed to set FRED API key: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to set FRED API key: {e}")
        return False


async def get_template_project_id(db: CloudSQLManager) -> str:
    """Get the main project ID"""
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM projects 
            WHERE is_template = true 
            AND name = 'main project'
            LIMIT 1
        """)
        
        if not row:
            raise Exception("main project not found")
        
        return str(row['id'])


async def get_feature_set_id(db: CloudSQLManager, project_id: str, name: str, set_type: str) -> str:
    """Get feature set ID by name and type"""
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM feature_sets
            WHERE project_id = $1 
            AND name = $2
            AND set_type = $3
        """, db._to_uuid(project_id), name, set_type)
        
        if not row:
            raise Exception(f"Feature set '{name}' ({set_type}) not found in project")
        
        return str(row['id'])


async def get_or_create_model(
    api_url: str,
    api_key: str,
    project_id: str,
    conditioning_set_id: str,
    target_set_id: str,
    model_name: str,
    model_description: str,
    db: CloudSQLManager,
) -> str:
    """Get existing model by name or create a new one"""
    # First, try to find existing model by name in the project
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM models
            WHERE project_id = $1 
            AND name = $2
            LIMIT 1
        """, db._to_uuid(project_id), model_name)
        
        if row:
            model_id = str(row['id'])
            logger.info(f"✅ Found existing model: {model_name} ({model_id})")
            return model_id
    
    # Model doesn't exist, create it
    url = f"{api_url}/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    payload = {
        "project_id": project_id,
        "conditioning_set_id": conditioning_set_id,
        "target_set_id": target_set_id,
        "name": model_name,
        "description": model_description,
    }
    
    logger.info(f"Creating new model: {model_name}")
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        model_id = result.get("id")
        logger.info(f"✅ Created model: {model_id}")
        return model_id


async def submit_training_job(
    api_url: str,
    api_key: str,
    model_id: str,
    training_start_date: str = "2015-01-01",
    training_end_date: str = "2025-09-30",
    n_regimes: int = 3,
    stride: int = 2,
    feature_grouping_threshold: float = 1.0,
    encoding_quality_weight: float = 100.0,
) -> str:
    """
    Submit a training job via API.
    
    Note: FRED API key will be automatically retrieved from the user's stored API keys.
    The system will use the user_id extracted from the API key authentication to fetch
    the stored FRED API key during data fetching.
    
    Args:
        feature_grouping_threshold: Correlation threshold for feature grouping (1.0 = no grouping, lower = enable grouping)
        encoding_quality_weight: Weight for reconstruction quality vs encoding components (higher = prioritize quality)
    """
    url = f"{api_url}/api/v1/ml/train"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    payload = {
        "model_id": model_id,
        "n_regimes": n_regimes,
        "compute_validation_ll": False,
        "feature_grouping_threshold": feature_grouping_threshold,
        "encoding_quality_weight": encoding_quality_weight,
        "data_fetch": {
            "start_date": training_start_date,
            "end_date": training_end_date,
            # Note: api_keys not needed - system automatically uses user's stored FRED API key
        },
        "sample": {
            "pastWindow": 100,  # Reduced from 252
            "futureWindow": 80,  # Increased from 63
            "stride": stride,
            "splitPercentages": {
                "training": 0.8,
                "validation": 0.1,
                "test": 0.1
            },
        },
    }
    
    logger.info(f"Submitting training job for model: {model_id}")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        job_id = result.get("job_id")
        logger.info(f"✅ Submitted training job: {job_id}")
        return job_id


async def get_training_status(
    api_url: str,
    api_key: str,
    model_id: str,
) -> Dict[str, Any]:
    """Get training job status for a model"""
    url = f"{api_url}/api/v1/ml/train/{model_id}/status"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def wait_for_training_completion(
    api_url: str,
    api_key: str,
    model_id: str,
    job_id: str,
    poll_interval: int = 30,
    max_wait_hours: int = 24,
) -> bool:
    """
    Poll training job status until completion or failure.
    
    Returns:
        True if training completed successfully, False if failed or timeout
    """
    max_wait_seconds = max_wait_hours * 3600
    elapsed = 0
    poll_count = 0
    terminal_states = {'completed', 'failed', 'training-failed'}
    
    logger.info(f"⏳ Waiting for training job {job_id[:8]}... to complete...")
    logger.info(f"   Polling every {poll_interval} seconds (max wait: {max_wait_hours} hours)")
    
    while elapsed < max_wait_seconds:
        try:
            status_data = await get_training_status(api_url, api_key, model_id)
            status = status_data.get("status", "unknown")
            current_job_id = status_data.get("job_id")
            message = status_data.get("message", "")
            
            # Verify we're tracking the right job
            if current_job_id and current_job_id != job_id:
                logger.warning(
                    f"⚠️  Job ID mismatch: expected {job_id[:8]}..., got {current_job_id[:8]}... "
                    f"(may be an old job, continuing to poll...)"
                )
            
            poll_count += 1
            status_line = f"   [Poll #{poll_count}] Status: {status} (elapsed: {elapsed//60}m {elapsed%60}s)"
            if message:
                status_line += f" - {message}"
            logger.info(status_line)
            
            # Check if job is in terminal state
            if status in terminal_states:
                if status == 'completed':
                    logger.info(f"✅ Training job completed successfully!")
                    return True
                else:
                    error_msg = status_data.get("error") or message
                    logger.error(f"❌ Training job failed with status: {status}")
                    if error_msg:
                        logger.error(f"   Error: {error_msg}")
                    return False
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Failed to get training status: {e}")
            if e.response.status_code == 404:
                logger.error(f"   Model {model_id} not found")
                return False
            # Retry after interval
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        except Exception as e:
            logger.error(f"❌ Error polling training status: {e}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
    
    logger.error(f"⏱️  Timeout: Training job did not complete within {max_wait_hours} hours")
    return False


async def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("Creating Models and Submitting Training Jobs")
    logger.info("=" * 80)
    
    if not API_KEY:
        logger.warning("⚠️  API_KEY not set. Using unauthenticated requests (may fail)")
    
    db = CloudSQLManager()
    
    try:
        # Step 1: Set FRED API key for user (if provided)
        if FRED_API_KEY:
            logger.info("\n🔑 Step 1: Setting FRED API key for user...")
            await set_user_fred_api_key(API_URL, API_KEY, FRED_API_KEY)
        else:
            logger.info("\n⚠️  Step 1: Skipping FRED API key setup (not provided)")
            logger.info("   Training will use user's existing stored FRED key if available")
        
        # Step 2: Get main project
        logger.info("\n📁 Step 2: Getting main project...")
        project_id = await get_template_project_id(db)
        logger.info(f"✅ Found project: {project_id}")
        
        # Step 3: Get or create models and submit training jobs sequentially
        logger.info(f"\n🎯 Step 3: Processing {len(MODEL_CONFIGS)} models and submitting training jobs...")
        logger.info("   Using date range: 2015-01-01 to 2025-09-30")
        logger.info("   Using windows: pastWindow=100, futureWindow=80")
        logger.info("   ⚠️  SEQUENTIAL MODE: Each job will complete before starting the next")
        logger.info("   This avoids connection pool exhaustion and FRED API rate limits")
        logger.info("   🔧 ICA DISABLED: Using PCA only for better reconstruction quality")
        logger.info("   🎯 Encoding quality weight: 100.0 (prioritizing R²)")
        logger.info("   📊 Feature grouping: Enabled (0.7) for yield/rate models, Disabled (1.0) for others")
        
        processed_models = []
        completed_jobs = []
        failed_jobs = []
        
        for i, config in enumerate(MODEL_CONFIGS, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"[{i}/{len(MODEL_CONFIGS)}] Processing: {config['model_name']}")
            logger.info(f"{'='*80}")
            try:
                # Get feature set IDs
                conditioning_set_id = await get_feature_set_id(
                    db, project_id, config['conditioning_set_name'], 'conditioning'
                )
                target_set_id = await get_feature_set_id(
                    db, project_id, config['target_set_name'], 'target'
                )
                
                logger.info(f"  Conditioning Set: {config['conditioning_set_name']} ({conditioning_set_id})")
                logger.info(f"  Target Set: {config['target_set_name']} ({target_set_id})")
                
                # Get or create model
                model_id = await get_or_create_model(
                    API_URL,
                    API_KEY,
                    project_id,
                    conditioning_set_id,
                    target_set_id,
                    config['model_name'],
                    config['model_description'],
                    db,
                )
                
                processed_models.append({
                    "name": config['model_name'],
                    "id": model_id,
                    "conditioning_set": config['conditioning_set_name'],
                    "target_set": config['target_set_name'],
                })
                
                # Check if model is already trained before submitting
                logger.info(f"\n🔍 Checking training status...")
                try:
                    status_data = await get_training_status(API_URL, API_KEY, model_id)
                    training_status = status_data.get("status", "idle")
                    existing_job_id = status_data.get("job_id")
                    
                    if training_status == "completed":
                        logger.info(f"  ✅ Model is already trained (status: completed)")
                        if existing_job_id:
                            logger.info(f"  📋 Previous job ID: {existing_job_id[:8]}...")
                        logger.info(f"  ⏭️  Skipping training for this model")
                        completed_jobs.append({
                            "model_name": config['model_name'],
                            "model_id": model_id,
                            "job_id": existing_job_id,
                            "skipped": True,
                        })
                        continue  # Skip to next model
                    elif training_status in {"failed", "training-failed"}:
                        logger.info(f"  ⚠️  Previous training failed (status: {training_status})")
                        if existing_job_id:
                            logger.info(f"  📋 Previous job ID: {existing_job_id[:8]}...")
                        logger.info(f"  🔄 Will retry training...")
                        # Continue to submit new job
                    elif training_status not in {"idle", "job-queued"}:
                        # Non-terminal state (e.g., fetching-data, generating-samples, etc.)
                        logger.warning(
                            f"  ⚠️  Model has non-terminal training status: {training_status}"
                        )
                        if existing_job_id:
                            logger.warning(
                                f"  📋 Active job ID: {existing_job_id[:8]}... "
                                f"(may be from previous run)"
                            )
                        logger.info(f"  🔄 Will attempt to submit new job (backend will handle conflicts)")
                        # Continue to submit - backend will reject if truly active
                    else:
                        logger.info(f"  ℹ️  Training status: {training_status} (ready for new job)")
                        # Continue to submit new job
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not check training status: {e}")
                    logger.info(f"  🔄 Proceeding with training job submission...")
                    # Continue anyway
                
                # Determine feature grouping threshold based on model type
                # Enable feature grouping for yield/rate models (lower threshold)
                # Disable for others (threshold = 1.0)
                conditioning_name = config['conditioning_set_name'].lower()
                has_yields = 'yield' in conditioning_name or 'rate' in conditioning_name or 'interest' in conditioning_name
                
                if has_yields:
                    feature_grouping_threshold = 0.7  # Enable grouping for yield/rate models
                    logger.info(f"  📊 Feature grouping ENABLED (threshold=0.7) for yield/rate model")
                else:
                    feature_grouping_threshold = 1.0  # No grouping for other models
                    logger.info(f"  📊 Feature grouping DISABLED (threshold=1.0)")
                
                # Higher encoding quality weight to push for better R²
                encoding_quality_weight = 100.0  # Increased from default 50.0 to prioritize reconstruction quality
                logger.info(f"  🎯 Encoding quality weight: {encoding_quality_weight} (prioritizing R²)")
                
                # Submit training job
                logger.info(f"\n📤 Submitting training job...")
                job_id = await submit_training_job(
                    API_URL,
                    API_KEY,
                    model_id,
                    training_start_date="2015-01-01",
                    training_end_date="2025-09-30",
                    feature_grouping_threshold=feature_grouping_threshold,
                    encoding_quality_weight=encoding_quality_weight,
                )
                
                # Wait for training to complete before moving to next job
                logger.info(f"\n⏳ Waiting for training job to complete before starting next job...")
                success = await wait_for_training_completion(
                    API_URL,
                    API_KEY,
                    model_id,
                    job_id,
                    poll_interval=30,  # Poll every 30 seconds
                    max_wait_hours=24,  # Max 24 hours per job
                )
                
                if success:
                    completed_jobs.append({
                        "model_name": config['model_name'],
                        "model_id": model_id,
                        "job_id": job_id,
                    })
                    logger.info(f"✅ Training completed successfully: {config['model_name']}")
                else:
                    failed_jobs.append({
                        "model_name": config['model_name'],
                        "model_id": model_id,
                        "job_id": job_id,
                    })
                    logger.error(f"❌ Training failed: {config['model_name']}")
                    logger.warning(f"⚠️  Continuing with next model...")
                
            except Exception as e:
                logger.error(f"❌ Failed to process '{config['model_name']}': {e}")
                import traceback
                traceback.print_exc()
                failed_jobs.append({
                    "model_name": config['model_name'],
                    "error": str(e),
                })
                continue
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ OPERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nProject ID: {project_id}")
        logger.info(f"Processed: {len(processed_models)} models")
        logger.info(f"Completed: {len(completed_jobs)} training jobs")
        logger.info(f"Failed: {len(failed_jobs)} training jobs")
        logger.info(f"Total: {len(processed_models)}/{len(MODEL_CONFIGS)}\n")
        
        if processed_models:
            logger.info("Processed models:")
            for idx, model_info in enumerate(processed_models, 1):
                logger.info(f"  {idx}. {model_info['name']}")
                logger.info(f"     - ID: {model_info['id']}")
                logger.info(f"     - Conditioning: {model_info['conditioning_set']}")
                logger.info(f"     - Target: {model_info['target_set']}")
        
        if completed_jobs:
            logger.info("\n✅ Completed training jobs:")
            for idx, job_info in enumerate(completed_jobs, 1):
                logger.info(f"  {idx}. {job_info['model_name']}")
                logger.info(f"     - Model ID: {job_info['model_id']}")
                if job_info.get('skipped'):
                    logger.info(f"     - Status: Skipped (already trained)")
                else:
                    logger.info(f"     - Job ID: {job_info['job_id']}")
        
        if failed_jobs:
            logger.info("\n❌ Failed training jobs:")
            for idx, job_info in enumerate(failed_jobs, 1):
                logger.info(f"  {idx}. {job_info['model_name']}")
                if 'job_id' in job_info:
                    logger.info(f"     - Model ID: {job_info['model_id']}")
                    logger.info(f"     - Job ID: {job_info['job_id']}")
                if 'error' in job_info:
                    logger.info(f"     - Error: {job_info['error']}")
        
        logger.info("\n📝 Next Steps:")
        if len(completed_jobs) < len(MODEL_CONFIGS):
            logger.info("  ⚠️  Some jobs failed. Check logs above for details.")
            logger.info("  You can re-run this script - it will skip completed models and retry failed ones.")
        else:
            logger.info("  ✅ All training jobs completed successfully!")
            logger.info("  Models are now ready for forecasting")
        logger.info("  Models are in the 'main project' template project")
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


