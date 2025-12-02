#!/usr/bin/env python3
"""Clear stale training job status from model metadata"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables
from dotenv import load_dotenv
env_path = backend_path / ".env"
if env_path.exists():
    load_dotenv(env_path)

from cloudsql_client import CloudSQLManager
import asyncio

async def clear_job_status(model_id: str, delete_samples: bool = False):
    """Clear training job status for a model"""
    db = CloudSQLManager()
    
    model = await db.get_model(model_id)
    if not model:
        print(f"❌ Model {model_id} not found")
        return
    
    user_id = model.get('user_id')
    metadata = model.get('metadata', {})
    training_meta = metadata.get('training', {})
    old_job_id = training_meta.get('job_id')
    old_status = training_meta.get('status', 'idle')
    
    print(f"📋 Current status:")
    print(f"   Job ID: {old_job_id}")
    print(f"   Status: {old_status}")
    
    # Clear the job status
    training_meta['job_id'] = None
    training_meta['status'] = 'idle'
    training_meta['message'] = None
    training_meta['error'] = None
    training_meta['started_at'] = None
    training_meta['finished_at'] = None
    training_meta['queued_at'] = None
    
    metadata['training'] = training_meta
    
    await db.update_model(model_id, {'metadata': metadata})
    
    print(f"✅ Cleared job status for model {model_id}")
    print(f"   Old job: {old_job_id}")
    print(f"   Old status: {old_status}")
    
    if delete_samples:
        print(f"\n🧹 Deleting all samples and encoding models for model {model_id}...")
        # Delete samples (this will cascade to encoded samples)
        await db.delete_samples_for_model(model_id, user_id)
        print(f"✅ Deleted all samples")
        
        # Delete encoding models from Cloud Storage
        from cloudsql_client import CloudStorageManager
        storage = CloudStorageManager()
        deleted_count = await storage.delete_encoders_for_model(user_id, model_id)
        print(f"✅ Deleted {deleted_count} encoding model files from Cloud Storage")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clear training job status")
    parser.add_argument('--model-id', type=str, required=True, help="Model ID to clear")
    parser.add_argument('--delete-samples', action='store_true', help="Also delete all samples and encoding models")
    args = parser.parse_args()
    
    asyncio.run(clear_job_status(args.model_id, delete_samples=args.delete_samples))

