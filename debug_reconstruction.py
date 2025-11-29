#!/usr/bin/env python3
"""
Debug script to visualize reconstruction quality using ReconstructionEngine.

This script:
1. Fetches a sample from the trained model
2. Uses ReconstructionEngine (same as production forecasting) to reconstruct
3. Compares original vs reconstructed to diagnose quality issues
4. Plots results for visual inspection

NEW ARCHITECTURE (Regime-Aware PCA/ICA):
- Encoding: Normalize → PCA/ICA per (feature_group, window_type, regime_id) → Store (with optional regime indicator)
- Decoding: Load → Extract regime indicator (if n_regimes > 1) → PCA/ICA inverse → Denormalize
"""

import asyncio
import json
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables from backend/.env
from dotenv import load_dotenv
env_path = backend_path / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from {env_path}")
else:
    print(f"⚠️  No .env file found at {env_path}")

from cloudsql_client import CloudSQLManager
from services.pipeline.reconstruct import ReconstructionEngine
import logging

# Suppress warnings from ReconstructionEngine about missing models (expected behavior)
# when clustering finds fewer regimes than metadata suggests
reconstruct_logger = logging.getLogger('services.pipeline.reconstruct')
reconstruct_logger.setLevel(logging.ERROR)  # Only show errors, not warnings about missing models


async def load_latest_model():
    """Find and return the latest trained model."""
    db = CloudSQLManager()
    
    print("🔍 Finding latest trained model...")
    async with db.connection() as conn:
        result = await conn.fetchrow("""
            SELECT id, name, user_id, created_at 
            FROM models 
            WHERE vine_copula_path IS NOT NULL
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        if not result:
            raise ValueError("No trained models found")
        
        model_id = str(result['id'])
        user_id = str(result['user_id'])
        print(f"✅ Found latest model: {result['name']} (created: {result['created_at']})")
        print(f"   Model ID: {model_id}")
        print(f"   User ID: {user_id}")
        
        return model_id, user_id


async def load_sample(model_id: str, user_id: str, sample_index: int = 0):
    """Load a normalized and encoded sample."""
    db = CloudSQLManager()
    
    # Get normalized samples
    samples_normalized = await db.get_samples_normalized(user_id, model_id, split_type='training')
    if not samples_normalized:
        raise ValueError(f"No normalized samples found for model {model_id}")
    
    print(f"✅ Found {len(samples_normalized)} normalized samples")
    
    # Get encoded samples
    samples_encoded = await db.get_samples_encoded(user_id, model_id, split_type='training')
    if not samples_encoded:
        raise ValueError(f"No encoded samples found for model {model_id}")
    
    print(f"✅ Found {len(samples_encoded)} encoded samples")
    
    # Pick a sample
    if sample_index >= len(samples_normalized):
        sample_index = 0
    
    sample_norm = samples_normalized[sample_index]
    sample_enc = samples_encoded[sample_index]
    
    print(f"\n📊 Using sample {sample_index}:")
    print(f"   Sample ID: {sample_norm.get('id') or sample_norm.get('sample_id', 'N/A')}")
    print(f"   Start date: {sample_norm.get('start_date')}")
    
    return sample_norm, sample_enc


def extract_feature_info(sample_enc: Dict) -> List[Dict[str, Any]]:
    """Extract feature_info from encoded sample metadata."""
    component_metadata = sample_enc.get('component_metadata', {})
    if isinstance(component_metadata, str):
        component_metadata = json.loads(component_metadata)
    
    feature_info_list = component_metadata.get('feature_info', [])
    print(f"\n📋 Found {len(feature_info_list)} feature/window combinations in encoded sample")
    
    return feature_info_list


async def reconstruct_window_with_engine(
    engine: ReconstructionEngine,
    sample_norm: Dict,
    sample_enc: Dict,
    feature_name: str,
    window_type: str,  # 'past', 'future_conditioning_series', 'future_target_residuals'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a window using ReconstructionEngine.
    
    Returns:
        tuple: (original_denormalized, reconstructed_denormalized)
    """
    # Map window_type to data keys and determine actual window_type for model lookup
    if window_type == 'past':
        normalized_key = 'normalized_past'
        encoded_key = 'encoded_past_series'
        order_key = 'feature_order_past'
        temporal_tag = 'past'
        data_type = 'encoded_normalized_series'
        # feature_info uses simplified data_type names
        feature_info_data_type = 'series'
        model_window_type = 'past'  # For model key lookup
    elif window_type == 'future_conditioning_series':
        normalized_key = 'normalized_future_conditioning_series'
        encoded_key = 'encoded_future_conditioning_series'
        order_key = 'feature_order_future_conditioning_series'
        temporal_tag = 'future'
        data_type = 'encoded_normalized_series'
        feature_info_data_type = 'series'
        model_window_type = 'future'  # For model key lookup
    elif window_type == 'future_target_residuals':
        normalized_key = 'normalized_future_target_residuals'
        encoded_key = 'encoded_future_target_residuals'
        order_key = 'feature_order_target'
        temporal_tag = 'future'
        data_type = 'encoded_normalized_residuals'
        feature_info_data_type = 'target_residuals'
        model_window_type = 'target_residuals'  # For model key lookup
    else:
        raise ValueError(f"Unknown window_type: {window_type}")
    
    print(f"\n🔍 Reconstructing {feature_name} ({window_type}):")
    
    # Get metadata
    metadata_norm = json.loads(sample_norm.get('metadata', '{}')) if isinstance(sample_norm.get('metadata'), str) else sample_norm.get('metadata', {})
    component_metadata = sample_enc.get('component_metadata', {})
    if isinstance(component_metadata, str):
        component_metadata = json.loads(component_metadata)
    
    # Get feature order (contains actual feature names, not group IDs)
    feature_order = metadata_norm.get(order_key, [])
    
    # Check if feature_name is a group ID or an actual feature name
    # Group IDs can be like "conditioning_group_Feature1_Feature2" or just the group name
    actual_features = None  # Will hold the list of feature names for groups
    group_id = None
    
    # Try to find the group in feature_groups
    if engine.feature_groups:
        for group in engine.feature_groups.get('conditioning_groups', []) + engine.feature_groups.get('target_groups', []):
            group_name = group.get('name')  # Use 'name' not 'id'
            if group_name == feature_name:
                group_id = group_name
                actual_features = group.get('features', [])
                break
    
    # If not found as group, check if it's a feature that belongs to a group
    if not actual_features and feature_name in engine.feature_to_group_map:
        group_id = engine.feature_to_group_map[feature_name]
        actual_features = engine._get_group_features(group_id)
    
    # If still not found, try to find encoding model with 3-tuple keys
    if not actual_features:
        # Try to find any model key matching this feature/group
        # Model keys are 3-tuples: (group_id, window_type, regime_id)
        # Use model_window_type (not window_type) for lookup
        matching_keys = [
            key for key in engine.encoding_models.keys()
            if isinstance(key, tuple) and len(key) == 3
            and key[0] == feature_name and key[1] == model_window_type
        ]
        
        if matching_keys:
            # Found a model - get features from the model metadata
            first_key = matching_keys[0]
            model_metadata = engine.encoding_models[first_key]
            actual_features = model_metadata.get('features', [])
            group_id = feature_name
            print(f"   ✓ Found encoding model for {feature_name}, extracted features: {actual_features}")
        
        if not actual_features:
            # Skip if this is a future_conditioning_series window and the group is not in the conditioning set
            if window_type == 'future_conditioning_series':
                print(f"   ⚠️  Skipping: Group/feature {feature_name} not in conditioning set for future windows")
                return None, None
            raise ValueError(f"Could not find encoding model or group for {feature_name} with temporal {temporal_tag}")
        
        # Use the first feature from the group to get normalized data
        # (we'll reconstruct the whole group, but we need one feature for comparison)
        if actual_features[0] not in feature_order:
            # Skip if this is a target_residuals window and the feature is not in the target set
            if window_type == 'future_target_residuals':
                print(f"   ⚠️  Skipping: Feature {actual_features[0]} (from group {group_id}) not in target set")
                return None, None
            raise ValueError(f"Feature {actual_features[0]} (from group {group_id}) not found in {order_key}")
        
        feature_idx = feature_order.index(actual_features[0])
        print(f"   Group {group_id} contains features: {actual_features}")
        print(f"   Using first feature '{actual_features[0]}' for comparison")
    else:
        # This is an actual feature name
        if feature_name not in feature_order:
            # Skip if this is a target_residuals window and the feature is not in the target set
            if window_type == 'future_target_residuals':
                print(f"   ⚠️  Skipping: Feature {feature_name} not in target set")
                return None, None
            # Skip if this is a future_conditioning_series window and the feature is not in the conditioning set
            if window_type == 'future_conditioning_series':
                print(f"   ⚠️  Skipping: Feature {feature_name} not in conditioning set for future windows")
                return None, None
            raise ValueError(f"Feature {feature_name} not found in {order_key}")
        
        feature_idx = feature_order.index(feature_name)
    
    # Get original normalized data
    normalized_data = sample_norm.get(normalized_key, [])
    if feature_idx >= len(normalized_data):
        raise ValueError(f"Feature index {feature_idx} out of range for {normalized_key}")
    
    original_normalized = np.array(normalized_data[feature_idx], dtype=float)
    print(f"   Original normalized shape: {original_normalized.shape}")
    print(f"   Original normalized range: [{np.min(original_normalized):.4f}, {np.max(original_normalized):.4f}]")
    
    # Get encoded data and slice for this feature
    encoded_full = sample_enc.get(encoded_key, [])
    
    # Convert to numpy array if it's a list
    if isinstance(encoded_full, list):
        encoded_full = np.array(encoded_full, dtype=float)
    elif not isinstance(encoded_full, np.ndarray):
        encoded_full = np.array(encoded_full, dtype=float)
    
    print(f"   Encoded full array length: {len(encoded_full)}")
    
    feature_info_list = component_metadata.get('feature_info', [])
    
    # Find the range for this feature or group
    feature_range = None
    matched_group_features = []
    
    matched_feature_info = None
    for info in feature_info_list:
        info_feature = info.get('feature', '')  # This is the group ID or feature name
        info_temporal = info.get('temporal_tag', '')
        info_data_type = info.get('data_type', '')
        group_features = info.get('group_features', [])
        
        # Match by temporal tag and data type (using simplified data_type from feature_info)
        temporal_match = (info_temporal == temporal_tag)
        data_type_match = (info_data_type == feature_info_data_type)
        
        if temporal_match and data_type_match:
            # Check if this matches our feature_name or group_id
            # Case 1: feature_name is a group ID (exact match with info_feature)
            # Case 2: feature_name is an actual feature (exact match with info_feature)
            # Case 3: group_id exists and matches info_feature
            # Note: group_features is empty in the metadata, so we match by info_feature
            feature_match = (info_feature == feature_name) or (group_id and info_feature == group_id)
            
            if feature_match:
                feature_range = info.get('range', [])
                matched_group_features = group_features if group_features else [info_feature]
                matched_feature_info = info  # Store for later use
                print(f"   ✓ Found encoded range: {feature_range}")
                print(f"     Feature/Group ID: {info_feature}")
                print(f"     Temporal: {info_temporal}, Data type: {info_data_type}")
                # Log regime info from feature_info if available
                info_n_regimes = info.get('n_regimes', None)
                info_has_regime_indicator = info.get('has_regime_indicator', None)
                if info_n_regimes is not None:
                    print(f"     Regime info from metadata: n_regimes={info_n_regimes}, has_indicator={info_has_regime_indicator}")
                break
    
    if feature_range is None or len(feature_range) != 2:
        raise ValueError(f"Could not find encoded range for {feature_name} in {window_type}")
    
    start, end = feature_range
    
    # IMPORTANT: The feature_info ranges are relative to the CONCATENATED training matrix:
    # [past_all, future_cond, target_residuals]
    # But we're slicing from individual arrays. We need to adjust the range based on which section
    # this feature belongs to.
    
    # Calculate offset based on which section this is in the training matrix
    # 1. Calculate the length of the past section
    encoded_past = sample_enc.get('encoded_past_series', [])
    if isinstance(encoded_past, list):
        encoded_past = np.array(encoded_past, dtype=float)
    past_len = len(encoded_past)
    
    # 2. Calculate the length of the future conditioning section
    encoded_future_cond = sample_enc.get('encoded_future_conditioning_series', [])
    if isinstance(encoded_future_cond, list):
        encoded_future_cond = np.array(encoded_future_cond, dtype=float)
    future_cond_len = len(encoded_future_cond)
    
    # 3. Determine which section this feature belongs to and adjust the range
    if window_type == 'past':
        # Past section: ranges start from 0, no offset needed
        adjusted_start = start
        adjusted_end = end
        section_name = "past"
    elif window_type == 'future_conditioning_series':
        # Future conditioning section: ranges start after past section
        adjusted_start = start - past_len
        adjusted_end = end - past_len
        section_name = "future_conditioning"
    elif window_type == 'future_target_residuals':
        # Target residuals section: ranges start after past + future_cond sections
        adjusted_start = start - past_len - future_cond_len
        adjusted_end = end - past_len - future_cond_len
        section_name = "target_residuals"
    else:
        adjusted_start = start
        adjusted_end = end
        section_name = "unknown"
    
    print(f"   Training matrix range: [{start}:{end}]")
    print(f"   Section: {section_name}, adjusted range: [{adjusted_start}:{adjusted_end}]")
    print(f"   Array length: {len(encoded_full)}")
    
    # Validate adjusted range
    if adjusted_start < 0 or adjusted_end > len(encoded_full):
        print(f"   ⚠️  WARNING: Adjusted range [{adjusted_start}:{adjusted_end}] is out of bounds for array of length {len(encoded_full)}")
        print(f"      This suggests a mismatch between feature_info ranges and actual array structure")
        # Try to use the original range if it fits
        if 0 <= start < len(encoded_full) and 0 <= end <= len(encoded_full):
            print(f"      Falling back to original range [{start}:{end}]")
            adjusted_start = start
            adjusted_end = end
        else:
            raise ValueError(f"Adjusted range [{adjusted_start}:{adjusted_end}] is out of bounds for {section_name} array (length={len(encoded_full)})")
    
    encoded_slice = encoded_full[adjusted_start:adjusted_end]
    print(f"   Encoded data slice: [{adjusted_start}:{adjusted_end}], length={len(encoded_slice)}")
    
    # Check if we need to handle regime indicator
    # If n_regimes > 1, the last coefficient is the regime indicator
    # We need to check how many regimes exist for this (group, window) combination
    n_regimes = 0
    lookup_group_id = group_id if group_id else feature_name
    available_regimes = []
    
    # First, try to get regime info from feature_info metadata (most reliable)
    if matched_feature_info:
        info_n_regimes = matched_feature_info.get('n_regimes', None)
        info_has_regime_indicator = matched_feature_info.get('has_regime_indicator', None)
        if info_n_regimes is not None:
            n_regimes = info_n_regimes
            print(f"   📋 Using regime info from feature_info metadata: n_regimes={n_regimes}")
    
    # Also find available regime IDs from encoding models (for verification)
    if lookup_group_id:
        # Find available regime IDs for this (group, window)
        # Use model_window_type for lookup
        available_regimes = [
            key[2] for key in engine.encoding_models.keys() 
            if isinstance(key, tuple) and len(key) == 3 
            and key[0] == lookup_group_id and key[1] == model_window_type
        ]
        # If we didn't get n_regimes from metadata, use model count
        if n_regimes == 0:
            n_regimes = len(available_regimes)
            print(f"   📋 Using regime info from encoding models: n_regimes={n_regimes}")
        
        # Verify consistency
        if matched_feature_info and matched_feature_info.get('n_regimes') is not None:
            if n_regimes != matched_feature_info.get('n_regimes'):
                print(f"   ⚠️  WARNING: Mismatch between feature_info n_regimes ({matched_feature_info.get('n_regimes')}) and model count ({n_regimes})")
    
    # Diagnostic: Extract and log regime indicator if present (for debugging)
    # Note: We don't strip it here - ReconstructionEngine will handle it
    regime_indicator_value = None
    actual_regime_id = None
    if n_regimes > 1 and len(encoded_slice) > 0:
        regime_indicator_value = float(encoded_slice[-1])
        # Round to nearest valid regime ID (same logic as ReconstructionEngine)
        actual_regime_id = int(min(available_regimes, key=lambda x: abs(x - regime_indicator_value)))
        print(f"   🔢 Regimes: {n_regimes} available {available_regimes}")
        print(f"   📌 Regime indicator in encoded data: {regime_indicator_value:.4f} → will use regime {actual_regime_id}")
        print(f"   📊 Encoded coefficients: {len(encoded_slice)} total ({len(encoded_slice)-1} PCA/ICA + 1 regime indicator)")
        
        # Verify the encoding model exists for this regime
        model_key = (lookup_group_id, model_window_type, actual_regime_id)
        if model_key in engine.encoding_models:
            model_info = engine.encoding_models[model_key]
            n_components = model_info.get('n_components', 'unknown')
            print(f"   ✅ Encoding model found: {model_key} (n_components={n_components})")
        else:
            print(f"   ⚠️  WARNING: Encoding model not found for {model_key}")
    else:
        print(f"   🔢 Regimes: {n_regimes} (single regime, no indicator)")
        if n_regimes == 1 and len(available_regimes) > 0:
            actual_regime_id = available_regimes[0]
            model_key = (lookup_group_id, model_window_type, actual_regime_id)
            if model_key in engine.encoding_models:
                model_info = engine.encoding_models[model_key]
                n_components = model_info.get('n_components', 'unknown')
                print(f"   ✅ Encoding model: {model_key} (n_components={n_components})")
    
    print(f"   📝 Note: ReconstructionEngine will extract regime indicator and select correct model")
    
    # Build window dict for ReconstructionEngine
    # Must include feature, temporal_tag, data_type, and the encoded data
    window_dict = {
        "feature": group_id if group_id else feature_name,  # Use group_id if found, otherwise feature_name
        "temporal_tag": temporal_tag,
        data_type: encoded_slice.tolist() if isinstance(encoded_slice, np.ndarray) else encoded_slice  # Convert to list if numpy array
    }
    
    # Extract reference values for target_residuals reconstruction
    # IMPORTANT: Reference values must be NORMALIZED (not denormalized) because:
    # - The decoded residuals are normalized
    # - We add normalized reference to normalized residuals to get normalized series
    # - Then we denormalize the series
    reference_values = {}
    if window_type == 'future_target_residuals':
        # Get past normalized data
        past_normalized = sample_norm.get('normalized_past', [])
        past_feature_order = metadata_norm.get('feature_order_past', [])
        
        # Get the features we're reconstructing
        features_to_reconstruct = actual_features if actual_features else [feature_name]
        
        for feat in features_to_reconstruct:
            if feat in past_feature_order:
                feat_idx = past_feature_order.index(feat)
                if feat_idx < len(past_normalized):
                    past_series = np.array(past_normalized[feat_idx], dtype=float)
                    if len(past_series) > 0:
                        # Get the last value - it's already normalized!
                        last_normalized = past_series[-1]
                        reference_values[feat] = float(last_normalized)
                        print(f"   Reference value (normalized) for {feat}: {last_normalized:.4f}")
            else:
                print(f"   ⚠️  No past data found for {feat}, cannot get reference value")
        
        # Also add reference value keyed by group_id if it exists (for group-based reconstruction)
        if group_id and features_to_reconstruct:
            # Use the first feature's reference value for the group
            first_feat = features_to_reconstruct[0]
            if first_feat in reference_values:
                reference_values[group_id] = reference_values[first_feat]
                print(f"   Reference value (normalized) for group {group_id}: {reference_values[first_feat]:.4f}")
    
    # Call ReconstructionEngine._reconstruct_single_window()
    # This handles: Extract regime indicator (if n_regimes > 1) → PCA/ICA inverse → Denormalize
    # For target_residuals, reference_values are needed to convert residuals to series
    reconstructed_list = await engine._reconstruct_single_window(window_dict, reference_values)
    
    if reconstructed_list is None or len(reconstructed_list) == 0:
        raise ValueError("ReconstructionEngine returned empty result")
    
    # Return all reconstructed features (for multivariate groups) or single feature (for univariate)
    # Build a list of (feature_name, original, reconstructed) tuples
    results = []
    
    if group_id and actual_features:
        # Multivariate group: return all features
        print(f"   📊 Reconstructing {len(actual_features)} features from group {feature_name}")
        
        for feat_name in actual_features:
            # Find reconstructed data for this feature
            reconstructed_denormalized = None
            for item in reconstructed_list:
                if item.get('feature') == feat_name:
                    reconstructed_denormalized = np.array(item.get('reconstructed_values', []), dtype=float)
                    break
            
            if reconstructed_denormalized is None:
                print(f"   ⚠️  Skipping {feat_name}: not found in reconstructed results")
                continue
            
            # Get original normalized data for this feature
            if feat_name not in feature_order:
                print(f"   ⚠️  Skipping {feat_name}: not found in {order_key}")
                continue
            
            feat_idx = feature_order.index(feat_name)
            if feat_idx >= len(normalized_data):
                print(f"   ⚠️  Skipping {feat_name}: index {feat_idx} out of range")
                continue
            
            feat_original_normalized = np.array(normalized_data[feat_idx], dtype=float)
            
            # For target_residuals, we need to convert residuals to series before denormalizing
            # The normalized_data contains normalized residuals, we add normalized reference value
            # to get normalized series, then denormalize
            if window_type == 'future_target_residuals':
                # Get reference value for this feature (already normalized)
                ref_value_normalized = reference_values.get(feat_name)
                if ref_value_normalized is None:
                    print(f"   ⚠️  No reference value for {feat_name}, cannot convert residuals to series")
                    continue
                
                # Convert normalized residuals to normalized series: series = reference + residuals
                # Both are already normalized, so just add them
                feat_original_normalized_series = feat_original_normalized + ref_value_normalized
                
                # Now denormalize the series
                norm_params = engine.normalization_params.get(feat_name, {})
                if norm_params:
                    mean = norm_params.get('mean', 0)
                    std = norm_params.get('std', 1)
                    feat_original_denormalized = feat_original_normalized_series * std + mean
                else:
                    feat_original_denormalized = feat_original_normalized_series
            else:
                # For series (past, future_conditioning_series), denormalize directly
                norm_params = engine.normalization_params.get(feat_name, {})
                if norm_params:
                    mean = norm_params.get('mean', 0)
                    std = norm_params.get('std', 1)
                    feat_original_denormalized = feat_original_normalized * std + mean
                else:
                    feat_original_denormalized = feat_original_normalized
            
            # Compute reconstruction error
            mse = np.mean((feat_original_denormalized - reconstructed_denormalized) ** 2)
            mae = np.mean(np.abs(feat_original_denormalized - reconstructed_denormalized))
            max_error = np.max(np.abs(feat_original_denormalized - reconstructed_denormalized))
            
            print(f"   ✓ {feat_name}:")
            print(f"      Reconstructed shape: {reconstructed_denormalized.shape}")
            print(f"      Original range: [{np.min(feat_original_denormalized):.4f}, {np.max(feat_original_denormalized):.4f}]")
            print(f"      Reconstructed range: [{np.min(reconstructed_denormalized):.4f}, {np.max(reconstructed_denormalized):.4f}]")
            print(f"      MSE: {mse:.6f}, MAE: {mae:.6f}, Max Error: {max_error:.6f}")
            
            results.append((feat_name, feat_original_denormalized, reconstructed_denormalized))
    else:
        # Univariate feature: return single feature
        lookup_feature = feature_name
        reconstructed_denormalized = None
        
        for item in reconstructed_list:
            if item.get('feature') == lookup_feature:
                reconstructed_denormalized = np.array(item.get('reconstructed_values', []), dtype=float)
                break
        
        if reconstructed_denormalized is None:
            print(f"   ⚠️  Available reconstructed features: {[item.get('feature') for item in reconstructed_list]}")
            raise ValueError(f"Could not find reconstructed data for {lookup_feature}")
        
        # For target_residuals, convert residuals to series before denormalizing
        if window_type == 'future_target_residuals':
            # Get reference value for this feature (already normalized)
            ref_value_normalized = reference_values.get(feature_name)
            if ref_value_normalized is None:
                print(f"   ⚠️  No reference value for {feature_name}, cannot convert residuals to series")
                raise ValueError(f"No reference value for {feature_name}")
            
            # Convert normalized residuals to normalized series: series = reference + residuals
            original_normalized_series = original_normalized + ref_value_normalized
            
            # Now denormalize the series
            norm_params = engine.normalization_params.get(feature_name, {})
            if norm_params:
                mean = norm_params.get('mean', 0)
                std = norm_params.get('std', 1)
                original_denormalized = original_normalized_series * std + mean
            else:
                original_denormalized = original_normalized_series
        else:
            # For series (past, future_conditioning_series), denormalize directly
            norm_params = engine.normalization_params.get(feature_name, {})
            if norm_params:
                mean = norm_params.get('mean', 0)
                std = norm_params.get('std', 1)
                original_denormalized = original_normalized * std + mean
            else:
                print(f"   ⚠️  No normalization params found for {feature_name}, using normalized data")
                original_denormalized = original_normalized
        
        # Compute error metrics
        mse = np.mean((original_denormalized - reconstructed_denormalized) ** 2)
        mae = np.mean(np.abs(original_denormalized - reconstructed_denormalized))
        max_error = np.max(np.abs(original_denormalized - reconstructed_denormalized))
        
        print(f"   Reconstructed shape: {reconstructed_denormalized.shape}")
        print(f"   Reconstructed range: [{np.min(reconstructed_denormalized):.4f}, {np.max(reconstructed_denormalized):.4f}]")
        print(f"   Original denormalized range: [{np.min(original_denormalized):.4f}, {np.max(original_denormalized):.4f}]")
        print(f"   📊 Reconstruction Error:")
        print(f"      MSE: {mse:.6f}")
        print(f"      MAE: {mae:.6f}")
        print(f"      Max Error: {max_error:.6f}")
        
        results.append((feature_name, original_denormalized, reconstructed_denormalized))
    
    return results


def plot_reconstruction(
    feature_name: str,
    window_type: str,
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_dir: Path,
):
    """Plot original vs reconstructed time series."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Original vs Reconstructed
    ax1 = axes[0]
    ax1.plot(original, label='Original', linewidth=2, alpha=0.8)
    ax1.plot(reconstructed, label='Reconstruction (PCA/ICA)', linewidth=2, alpha=0.8, linestyle='--')
    ax1.set_title(f'{feature_name} ({window_type}): Original vs Reconstructed')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Error
    ax2 = axes[1]
    error = original - reconstructed
    ax2.plot(error, label='Error (Original - Reconstructed)', color='red', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Reconstruction Error')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add error statistics
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    stats_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nMax Error: {max_error:.6f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
    filename = output_dir / f"{safe_feature_name}_{window_type}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved plot to {filename}")


async def main():
    parser = argparse.ArgumentParser(description="Debug reconstruction quality using ReconstructionEngine")
    parser.add_argument('--model-id', type=str, help="Model ID (default: latest trained model)")
    parser.add_argument('--sample-idx', type=int, default=0, help="Sample index to use (default: 0)")
    parser.add_argument('--feature', type=str, help="Specific feature to reconstruct (default: all)")
    parser.add_argument('--window', type=str, choices=['past', 'future_conditioning_series', 'future_target_residuals'],
                       help="Specific window to reconstruct (default: all)")
    parser.add_argument('--output-dir', type=str, default='reconstruction_plots',
                       help="Output directory for plots (default: reconstruction_plots)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("🔬 RECONSTRUCTION DEBUGGER (Using ReconstructionEngine)")
    print("=" * 80)
    
    # Load model
    if args.model_id:
        model_id = args.model_id
        # Get user_id from model
        db = CloudSQLManager()
        model = await db.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        user_id = str(model['user_id'])
    else:
        model_id, user_id = await load_latest_model()
    
    # Initialize ReconstructionEngine
    print("\n🔧 Initializing ReconstructionEngine...")
    print("   (Note: Missing model warnings are expected if clustering found fewer regimes)")
    
    # Suppress warnings during initialization (missing models are expected)
    import warnings
    import logging
    
    # Temporarily set reconstruct logger to ERROR level to suppress warnings
    reconstruct_logger = logging.getLogger('services.pipeline.reconstruct')
    original_level = reconstruct_logger.level
    reconstruct_logger.setLevel(logging.ERROR)
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            engine = ReconstructionEngine(model_id, user_id)
            await engine.initialize()
    finally:
        # Restore original log level
        reconstruct_logger.setLevel(original_level)
    
    print(f"   ✅ Initialized with {len(engine.encoding_models)} encoding models")
    
    # Load feature groups (needed to resolve group IDs to individual feature names)
    engine.feature_groups = await engine.get_feature_groups()
    if engine.feature_groups:
        # Build feature_to_group_map for reverse lookup
        # Use 'name' not 'id' for group identification
        for group in engine.feature_groups.get('conditioning_groups', []) + engine.feature_groups.get('target_groups', []):
            group_name = group.get('name')  # Use 'name' not 'id'
            for feature in group.get('features', []):
                engine.feature_to_group_map[feature] = group_name
        print(f"   ✓ Loaded {len(engine.feature_groups.get('conditioning_groups', []))} conditioning groups and {len(engine.feature_groups.get('target_groups', []))} target groups")
    
    print(f"   ✓ Loaded {len(engine.encoding_models)} encoding models (3-tuple keys: group_id, window_type, regime_id)")
    print(f"   ✓ Loaded {len(engine.normalization_params)} normalization params")
    
    # Load sample
    sample_norm, sample_enc = await load_sample(model_id, user_id, args.sample_idx)
    
    # Extract feature info
    feature_info_list = extract_feature_info(sample_enc)
    
    # Build list of windows to process
    windows_to_process = []
    
    # Get all unique features from feature_info
    all_features = set()
    for info in feature_info_list:
        feature_id = info.get('feature', '')
        group_features = info.get('group_features', [])
        
        # Add all features from the group
        if group_features:
            all_features.update(group_features)
        else:
            all_features.add(feature_id)
    
    print(f"\n📋 Available features: {sorted(all_features)}")
    
    # Determine which features and windows to process
    features_to_process = [args.feature] if args.feature else sorted(all_features)
    windows_to_process_types = [args.window] if args.window else ['past', 'future_conditioning_series', 'future_target_residuals']
    
    print(f"\n📊 Will process {len(features_to_process)} features × {len(windows_to_process_types)} windows = {len(features_to_process) * len(windows_to_process_types)} combinations")
    
    # Process each combination
    success_count = 0
    error_count = 0
    
    for feature in features_to_process:
        for window_type in windows_to_process_types:
            try:
                print(f"\n{'='*60}")
                results = await reconstruct_window_with_engine(
                    engine=engine,
                    sample_norm=sample_norm,
                    sample_enc=sample_enc,
                    feature_name=feature,
                    window_type=window_type
                )
                
                # Skip if the feature/window combination doesn't exist
                if results is None or (isinstance(results, tuple) and results[0] is None):
                    continue
                
                # results is a list of (feature_name, original, reconstructed) tuples
                for feat_name, original, reconstructed in results:
                    plot_reconstruction(
                        feature_name=feat_name,
                        window_type=window_type,
                        original=original,
                        reconstructed=reconstructed,
                        output_dir=output_dir
                    )
                
                success_count += len(results)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
    
    print(f"\n{'='*80}")
    print(f"✅ Done! {success_count} successful, {error_count} errors")
    print(f"📁 Plots saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
