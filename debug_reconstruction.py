#!/usr/bin/env python3
"""
Debug script to visualize wavelet + PCA reconstruction quality.

This script:
1. Fetches a sample from the trained model
2. Takes a specific feature/window
3. Reconstructs it through the full pipeline (wavelet -> PCA -> inverse PCA -> inverse wavelet)
4. Plots original vs reconstructed to diagnose quality issues
"""

import asyncio
import json
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

from src.cloudsql_client import CloudSQLManager, CloudStorageManager
from src.wavelet_encoder import WaveletEncoder
from src.pca_ica_encoder import PCAICAEncoder


async def load_sample_and_encoders(model_id: str, sample_index: int = 0):
    """Load a sample and its encoding models."""
    db = CloudSQLManager()
    gcs = CloudStorageManager()
    
    # Get model info
    model = await db.get_model(model_id)
    if not model:
        raise ValueError(f"Model {model_id} not found")
    
    user_id = model['user_id']
    print(f"📦 Model: {model['name']}")
    print(f"   Owner: {user_id}")
    
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
    
    # Load wavelet config from model metadata
    model_metadata = json.loads(model.get('metadata', '{}')) if isinstance(model.get('metadata'), str) else model.get('metadata', {})
    wavelet_config = model_metadata.get('wavelet_config', {})
    
    if not wavelet_config:
        raise ValueError("No wavelet config found in model metadata")
    
    print(f"\n🌊 Wavelet encoders available: {len(wavelet_config)} features")
    
    # Load PCA/ICA models metadata from database
    encoding_models = await db.load_encoding_models_metadata(model_id)
    print(f"📦 PCA/ICA models available: {len(encoding_models)}")
    
    pca_models = {}
    for model_key, model_metadata_entry in encoding_models.items():
        gcs_path = model_metadata_entry.get('storage_path')
        if gcs_path:
            try:
                encoder = await gcs.download_encoder(gcs_path)
                pca_models[model_key] = encoder
                print(f"   ✓ Loaded {model_key}")
            except Exception as e:
                print(f"   ✗ Failed to load {model_key}: {e}")
    
    # Load feature groups
    conditioning_set_id = model.get('conditioning_set_id')
    target_set_id = model.get('target_set_id')
    
    feature_groups = {}
    if conditioning_set_id:
        cond_set = await db.get_feature_set(conditioning_set_id)
        if cond_set:
            cond_groups = json.loads(cond_set.get('feature_groups', '{}')) if isinstance(cond_set.get('feature_groups'), str) else cond_set.get('feature_groups', {})
            feature_groups['conditioning_groups'] = cond_groups.get('groups', [])
    
    if target_set_id:
        target_set = await db.get_feature_set(target_set_id)
        if target_set:
            target_groups = json.loads(target_set.get('feature_groups', '{}')) if isinstance(target_set.get('feature_groups'), str) else target_set.get('feature_groups', {})
            feature_groups['target_groups'] = target_groups.get('groups', [])
    
    print(f"\n📦 Feature groups:")
    print(f"   Conditioning: {len(feature_groups.get('conditioning_groups', []))} groups")
    print(f"   Target: {len(feature_groups.get('target_groups', []))} groups")
    
    return {
        'model': model,
        'sample_norm': sample_norm,
        'sample_enc': sample_enc,
        'wavelet_config': wavelet_config,
        'pca_models': pca_models,
        'feature_groups': feature_groups,
    }


def reconstruct_window(
    feature_name: str,
    window_type: str,  # 'past', 'future_conditioning_series', 'future_target_residuals'
    normalized_data: np.ndarray,
    encoded_data: np.ndarray,
    wavelet_config: dict,
    pca_models: dict,
    feature_groups: dict,
    metadata: dict,
):
    """
    Reconstruct a window through the full pipeline.
    
    Returns:
        tuple: (original_normalized, reconstructed_from_encoded)
    """
    # Determine data keys
    if window_type == 'past':
        data_key = 'normalized_past'
        encoded_key = 'encoded_past_series'
        order_key = 'feature_order_past'
        temporal_tag = 'past'
    elif window_type == 'future_conditioning_series':
        data_key = 'normalized_future_conditioning_series'
        encoded_key = 'encoded_future_conditioning_series'
        order_key = 'feature_order_future_conditioning_series'
        temporal_tag = 'future'
    elif window_type == 'future_target_residuals':
        data_key = 'normalized_future_target_residuals'
        encoded_key = 'encoded_future_target_residuals'
        order_key = 'feature_order_target'
        temporal_tag = 'future'
    else:
        raise ValueError(f"Unknown window_type: {window_type}")
    
    # Get feature order
    feature_order = metadata.get(order_key, [])
    if feature_name not in feature_order:
        raise ValueError(f"Feature {feature_name} not found in {order_key}")
    
    feature_idx = feature_order.index(feature_name)
    
    # Get original normalized data
    original = normalized_data[feature_idx]
    
    print(f"\n🔍 Reconstructing {feature_name} ({window_type}):")
    print(f"   Original shape: {original.shape}")
    print(f"   Original range: [{np.min(original):.4f}, {np.max(original):.4f}]")
    
    # Get the encoded data slice for this specific feature/group from metadata
    # The encoded_data passed in is the ENTIRE window array, we need to slice it
    component_metadata = metadata.get('component_metadata', {})
    feature_info_list = component_metadata.get('feature_info', [])
    
    # Find this feature's range in the encoded data
    feature_encoded_range = None
    for info in feature_info_list:
        info_feature = info.get('feature', '')
        info_temporal = info.get('temporal_tag', '')
        info_data_type = info.get('data_type', '')
        
        # Match by feature name and window type
        matches_feature = (info_feature == feature_name or 
                          (info_feature.startswith('group_') and feature_name in info.get('group_features', [])))
        matches_temporal = (temporal_tag in info_temporal)
        matches_data_type = ('residuals' in window_type) == ('residuals' in info_data_type or 'residuals' in info_temporal)
        
        if matches_feature and matches_temporal and matches_data_type:
            feature_encoded_range = info.get('range', [])
            print(f"   Found metadata range: {feature_encoded_range}")
            break
    
    # Slice the encoded data to get just this feature/group's portion
    if feature_encoded_range and len(feature_encoded_range) == 2:
        start, end = feature_encoded_range
        encoded_data_slice = encoded_data[start:end]
        print(f"   Sliced encoded data from [{start}:{end}], length: {len(encoded_data_slice)}")
    else:
        print(f"   ⚠️  No metadata range found, using full encoded data (length: {len(encoded_data)})")
        encoded_data_slice = encoded_data
    
    # Step 1: Get wavelet encoder
    wavelet_key = f"{feature_name}_{temporal_tag}"
    if temporal_tag == 'future' and 'residuals' in window_type:
        wavelet_key += "_residuals"
    
    if feature_name not in wavelet_config:
        raise ValueError(f"No wavelet config for {feature_name}")
    
    window_label = temporal_tag if 'residuals' not in window_type else f"{temporal_tag}_residuals"
    feature_wavelet_config = wavelet_config[feature_name].get(window_label)
    if not feature_wavelet_config:
        raise ValueError(f"No wavelet config for {feature_name}.{window_label}")
    
    wavelet_encoder = WaveletEncoder.from_config(feature_wavelet_config)
    print(f"   Wavelet: {wavelet_encoder.wavelet_family}, level={wavelet_encoder.fitted_level}, n_coeffs={wavelet_encoder.n_coefficients}")
    
    # Step 2: Apply wavelet transform to original
    wavelet_coeffs_original = wavelet_encoder.encode(original)
    print(f"   Wavelet coeffs from original: {wavelet_coeffs_original.shape}")
    print(f"   Wavelet coeffs range: [{np.min(wavelet_coeffs_original):.4f}, {np.max(wavelet_coeffs_original):.4f}]")
    
    # Step 3: Check if feature is in a multivariate group (needs PCA)
    is_multivariate = False
    group_id = None
    pca_model = None
    
    # Check conditioning groups
    for group in feature_groups.get('conditioning_groups', []):
        if feature_name in group.get('features', []):
            is_multivariate = group.get('is_multivariate', len(group.get('features', [])) > 1)
            group_id = group.get('id') or group.get('name')
            break
    
    # Check target groups
    if not is_multivariate:
        for group in feature_groups.get('target_groups', []):
            if feature_name in group.get('features', []):
                is_multivariate = group.get('is_multivariate', len(group.get('features', [])) > 1)
                group_id = group.get('id') or group.get('name')
                break
    
    if is_multivariate and group_id:
        # Look for PCA model
        # group_id already has "group_" prefix for PCA-based groups
        if not group_id.startswith('group_'):
            pca_key = f"group_{group_id}_{temporal_tag}"
        else:
            pca_key = f"{group_id}_{temporal_tag}"
        if 'residuals' in window_type:
            pca_key += "_residuals"
        
        pca_model = pca_models.get(pca_key)
        if pca_model:
            print(f"   PCA model: {pca_key}")
            print(f"   PCA components: {pca_model.n_components_}")
        else:
            print(f"   ⚠️  No PCA model found for {pca_key} (expected for multivariate group)")
            is_multivariate = False
    
    # Step 4: Reconstruct from encoded data
    if is_multivariate and pca_model:
        # Multivariate: PCA inverse + Wavelet inverse
        print(f"   🔄 Multivariate reconstruction (PCA + Wavelet)")
        
        # The encoded data is stored as a flat array of PCA components
        # We need to:
        # 1. Extract the PCA components for this group/window
        # 2. Apply PCA inverse to get wavelet coefficients for all features in the group
        # 3. Extract this specific feature's wavelet coefficients
        # 4. Apply wavelet inverse to get the time series
        
        # Get the group's features
        group_features = []
        for group in feature_groups.get('conditioning_groups', []) + feature_groups.get('target_groups', []):
            group_id_check = group.get('id') or group.get('name')
            if group_id_check == group_id:
                group_features = group.get('features', [])
                break
        
        if not group_features:
            print(f"   ⚠️  Could not find group features for {group_id}")
            reconstructed = wavelet_encoder.decode(wavelet_coeffs_original)
        else:
            print(f"   Group features: {group_features}")
            
            # The encoded data is a flat array
            # For multivariate groups, the PCA components are stored per wavelet coefficient
            # Structure: [pca_comp_0_for_coeff_0, pca_comp_1_for_coeff_0, ..., pca_comp_0_for_coeff_1, ...]
            
            n_features_in_group = len(group_features)
            n_pca_components = pca_model.n_components_
            n_wavelet_coeffs = wavelet_encoder.n_coefficients
            
            print(f"   Expected structure: {n_wavelet_coeffs} coeffs × {n_pca_components} PCA components = {n_wavelet_coeffs * n_pca_components} values")
            print(f"   Encoded data slice length: {len(encoded_data_slice)}")
            
            # Reshape encoded data: (n_coeffs, n_components)
            try:
                encoded_matrix = np.array(encoded_data_slice).reshape(n_wavelet_coeffs, n_pca_components)
                print(f"   Reshaped encoded data: {encoded_matrix.shape}")
                print(f"   Encoded data range: [{np.min(encoded_matrix):.4f}, {np.max(encoded_matrix):.4f}]")
                
                # Apply PCA inverse per wavelet coefficient
                # This gives us (n_coeffs, n_features)
                decoded_wavelet_matrix = pca_model.decode(encoded_matrix)
                print(f"   PCA decoded shape: {decoded_wavelet_matrix.shape}")
                print(f"   PCA decoded range: [{np.min(decoded_wavelet_matrix):.4f}, {np.max(decoded_wavelet_matrix):.4f}]")
                
                # Extract this feature's wavelet coefficients
                if feature_name not in group_features:
                    print(f"   ⚠️  Feature {feature_name} not in group features!")
                    reconstructed = wavelet_encoder.decode(wavelet_coeffs_original)
                else:
                    feature_idx_in_group = group_features.index(feature_name)
                    wavelet_coeffs_from_pca = decoded_wavelet_matrix[:, feature_idx_in_group]
                    
                    print(f"   Feature index in group: {feature_idx_in_group}")
                    print(f"   Wavelet coeffs from PCA: {wavelet_coeffs_from_pca.shape}")
                    print(f"   Wavelet coeffs from PCA range: [{np.min(wavelet_coeffs_from_pca):.4f}, {np.max(wavelet_coeffs_from_pca):.4f}]")
                    
                    # Compare with original wavelet coeffs
                    coeff_mse = np.mean((wavelet_coeffs_original - wavelet_coeffs_from_pca) ** 2)
                    print(f"   📊 Wavelet coeff MSE (original vs PCA-decoded): {coeff_mse:.6f}")
                    
                    # Apply wavelet inverse
                    reconstructed = wavelet_encoder.decode(wavelet_coeffs_from_pca)
                    
            except Exception as e:
                print(f"   ⚠️  Error during PCA reconstruction: {e}")
                print(f"   Falling back to wavelet-only reconstruction")
                reconstructed = wavelet_encoder.decode(wavelet_coeffs_original)
    else:
        # Univariate: just wavelet
        print(f"   🔄 Univariate reconstruction (Wavelet only)")
        reconstructed = wavelet_encoder.decode(wavelet_coeffs_original)
    
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Reconstructed range: [{np.min(reconstructed):.4f}, {np.max(reconstructed):.4f}]")
    
    # Calculate reconstruction error
    if len(reconstructed) == len(original):
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        print(f"   📊 MSE: {mse:.6f}, MAE: {mae:.6f}")
    else:
        print(f"   ⚠️  Length mismatch: original={len(original)}, reconstructed={len(reconstructed)}")
    
    # Also reconstruct using wavelet-only (for comparison)
    reconstructed_wavelet_only = wavelet_encoder.decode(wavelet_coeffs_original)
    
    return original, reconstructed, reconstructed_wavelet_only, wavelet_coeffs_original


def plot_reconstruction(feature_name: str, window_type: str, original: np.ndarray, reconstructed: np.ndarray, 
                        reconstructed_wavelet_only: np.ndarray, wavelet_coeffs: np.ndarray):
    """Plot original vs reconstructed."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    ax1 = axes[0]
    time_steps = np.arange(len(original))
    
    ax1.plot(time_steps, original, 'b-', linewidth=2.5, label='Original (Normalized)', alpha=0.8)
    ax1.plot(time_steps, reconstructed_wavelet_only, 'g--', linewidth=2, label='Wavelet-Only Reconstruction', alpha=0.7)
    ax1.plot(time_steps, reconstructed, 'r--', linewidth=2, label='PCA+Wavelet Reconstruction', alpha=0.7)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Normalized Value', fontsize=12)
    ax1.set_title(f'{feature_name} - {window_type}\nOriginal vs Reconstructed', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display metrics for both reconstructions
    metrics_lines = []
    
    if len(reconstructed_wavelet_only) == len(original):
        mse_wav = np.mean((original - reconstructed_wavelet_only) ** 2)
        mae_wav = np.mean(np.abs(original - reconstructed_wavelet_only))
        corr_wav = np.corrcoef(original, reconstructed_wavelet_only)[0, 1]
        metrics_lines.append(f'Wavelet-Only:')
        metrics_lines.append(f'  MSE: {mse_wav:.6f}')
        metrics_lines.append(f'  MAE: {mae_wav:.6f}')
        metrics_lines.append(f'  Corr: {corr_wav:.4f}')
    
    if len(reconstructed) == len(original):
        mse_pca = np.mean((original - reconstructed) ** 2)
        mae_pca = np.mean(np.abs(original - reconstructed))
        corr_pca = np.corrcoef(original, reconstructed)[0, 1]
        metrics_lines.append(f'PCA+Wavelet:')
        metrics_lines.append(f'  MSE: {mse_pca:.6f}')
        metrics_lines.append(f'  MAE: {mae_pca:.6f}')
        metrics_lines.append(f'  Corr: {corr_pca:.4f}')
    
    if metrics_lines:
        metrics_text = '\n'.join(metrics_lines)
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
    
    # Plot 2: Wavelet coefficients
    ax2 = axes[1]
    coeff_indices = np.arange(len(wavelet_coeffs))
    
    ax2.stem(coeff_indices, wavelet_coeffs, basefmt=' ', linefmt='g-', markerfmt='go')
    ax2.set_xlabel('Coefficient Index', fontsize=12)
    ax2.set_ylabel('Coefficient Value', fontsize=12)
    ax2.set_title(f'Wavelet Coefficients (cA only, n={len(wavelet_coeffs)})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add coefficient statistics
    coeff_stats = f'Mean: {np.mean(wavelet_coeffs):.4f}\nStd: {np.std(wavelet_coeffs):.4f}\nMax: {np.max(np.abs(wavelet_coeffs)):.4f}'
    ax2.text(0.98, 0.98, coeff_stats, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"reconstruction_debug_{feature_name.replace(' ', '_')}_{window_type}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved plot to: {filename}")
    
    plt.show()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Debug wavelet + PCA reconstruction quality')
    parser.add_argument('--model-id', type=str, default="372d48f3-b9c0-4f5e-b7d0-547a2505386a",
                        help='Model ID to analyze')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index to use (0-based)')
    parser.add_argument('--feature', type=str, default=None,
                        help='Specific feature to analyze (if not provided, analyzes all)')
    parser.add_argument('--window-type', type=str, default=None,
                        choices=['past', 'future_conditioning_series', 'future_target_residuals'],
                        help='Specific window type to analyze (if not provided, analyzes all)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 WAVELET + PCA RECONSTRUCTION DEBUGGER")
    print("=" * 80)
    
    # Load data
    data = await load_sample_and_encoders(args.model_id, args.sample_idx)
    
    # Get sample metadata
    sample_norm = data['sample_norm']
    sample_enc = data['sample_enc']
    metadata = json.loads(sample_norm.get('metadata', '{}')) if isinstance(sample_norm.get('metadata'), str) else sample_norm.get('metadata', {})
    
    # Add component_metadata from encoded sample (this contains the ranges for slicing)
    component_metadata = json.loads(sample_enc.get('component_metadata', '{}')) if isinstance(sample_enc.get('component_metadata'), str) else sample_enc.get('component_metadata', {})
    metadata['component_metadata'] = component_metadata
    
    # Determine which features and windows to process
    window_configs = []
    
    if args.feature and args.window_type:
        # Single feature/window
        window_configs.append((args.feature, args.window_type))
    elif args.window_type:
        # All features for a specific window type
        order_key = {
            'past': 'feature_order_past',
            'future_conditioning_series': 'feature_order_future_conditioning_series',
            'future_target_residuals': 'feature_order_target'
        }[args.window_type]
        features = metadata.get(order_key, [])
        window_configs = [(feat, args.window_type) for feat in features]
    elif args.feature:
        # Single feature, all window types
        for window_type in ['past', 'future_conditioning_series', 'future_target_residuals']:
            order_key = {
                'past': 'feature_order_past',
                'future_conditioning_series': 'feature_order_future_conditioning_series',
                'future_target_residuals': 'feature_order_target'
            }[window_type]
            features = metadata.get(order_key, [])
            if args.feature in features:
                window_configs.append((args.feature, window_type))
    else:
        # All features, all window types
        for window_type in ['past', 'future_conditioning_series', 'future_target_residuals']:
            order_key = {
                'past': 'feature_order_past',
                'future_conditioning_series': 'feature_order_future_conditioning_series',
                'future_target_residuals': 'feature_order_target'
            }[window_type]
            features = metadata.get(order_key, [])
            window_configs.extend([(feat, window_type) for feat in features])
    
    print(f"\n📊 Processing {len(window_configs)} feature/window combinations...")
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each feature/window
    results = []
    for i, (feature_name, window_type) in enumerate(window_configs, 1):
        print(f"\n[{i}/{len(window_configs)}] Processing {feature_name} ({window_type})")
        
        try:
            # Get normalized and encoded data
            data_key_map = {
                'past': ('normalized_past', 'encoded_past_series'),
                'future_conditioning_series': ('normalized_future_conditioning_series', 'encoded_future_conditioning_series'),
                'future_target_residuals': ('normalized_future_target_residuals', 'encoded_future_target_residuals')
            }
            norm_key, enc_key = data_key_map[window_type]
            normalized_data = sample_norm.get(norm_key, [])
            encoded_data = sample_enc.get(enc_key, [])
            
            # Reconstruct
            original, reconstructed, reconstructed_wavelet_only, wavelet_coeffs = reconstruct_window(
                feature_name=feature_name,
                window_type=window_type,
                normalized_data=normalized_data,
                encoded_data=encoded_data,
                wavelet_config=data['wavelet_config'],
                pca_models=data['pca_models'],
                feature_groups=data['feature_groups'],
                metadata=metadata,
            )
            
            # Calculate metrics
            if len(reconstructed) == len(original):
                mse_pca = np.mean((original - reconstructed) ** 2)
                mae_pca = np.mean(np.abs(original - reconstructed))
                corr_pca = np.corrcoef(original, reconstructed)[0, 1]
            else:
                mse_pca = mae_pca = corr_pca = np.nan
            
            if len(reconstructed_wavelet_only) == len(original):
                mse_wav = np.mean((original - reconstructed_wavelet_only) ** 2)
                mae_wav = np.mean(np.abs(original - reconstructed_wavelet_only))
                corr_wav = np.corrcoef(original, reconstructed_wavelet_only)[0, 1]
            else:
                mse_wav = mae_wav = corr_wav = np.nan
            
            results.append({
                'feature': feature_name,
                'window_type': window_type,
                'mse_wavelet': mse_wav,
                'mae_wavelet': mae_wav,
                'corr_wavelet': corr_wav,
                'mse_pca_wavelet': mse_pca,
                'mae_pca_wavelet': mae_pca,
                'corr_pca_wavelet': corr_pca,
            })
            
            # Plot (save to output directory)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = output_dir / f"reconstruction_{feature_name.replace(' ', '_')}_{window_type}_{timestamp}.png"
            
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Time series comparison
            ax1 = axes[0]
            time_steps = np.arange(len(original))
            
            ax1.plot(time_steps, original, 'b-', linewidth=2.5, label='Original (Normalized)', alpha=0.8)
            ax1.plot(time_steps, reconstructed_wavelet_only, 'g--', linewidth=2, label='Wavelet-Only', alpha=0.7)
            ax1.plot(time_steps, reconstructed, 'r--', linewidth=2, label='PCA+Wavelet', alpha=0.7)
            
            ax1.set_xlabel('Time Step', fontsize=12)
            ax1.set_ylabel('Normalized Value', fontsize=12)
            ax1.set_title(f'{feature_name} - {window_type}\nOriginal vs Reconstructed', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Add metrics
            metrics_text = f'Wavelet-Only:\n  MSE: {mse_wav:.6f}\n  MAE: {mae_wav:.6f}\n  Corr: {corr_wav:.4f}\n'
            metrics_text += f'PCA+Wavelet:\n  MSE: {mse_pca:.6f}\n  MAE: {mae_pca:.6f}\n  Corr: {corr_pca:.4f}'
            ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
            
            # Plot 2: Wavelet coefficients
            ax2 = axes[1]
            coeff_indices = np.arange(len(wavelet_coeffs))
            
            ax2.stem(coeff_indices, wavelet_coeffs, basefmt=' ', linefmt='g-', markerfmt='go')
            ax2.set_xlabel('Coefficient Index', fontsize=12)
            ax2.set_ylabel('Coefficient Value', fontsize=12)
            ax2.set_title(f'Wavelet Coefficients (cA only, n={len(wavelet_coeffs)})', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            coeff_stats = f'Mean: {np.mean(wavelet_coeffs):.4f}\nStd: {np.std(wavelet_coeffs):.4f}\nMax: {np.max(np.abs(wavelet_coeffs)):.4f}'
            ax2.text(0.98, 0.98, coeff_stats, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Saved to {filename}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append({
                'feature': feature_name,
                'window_type': window_type,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 RECONSTRUCTION QUALITY SUMMARY")
    print("=" * 80)
    
    for result in results:
        if 'error' in result:
            print(f"\n❌ {result['feature']} ({result['window_type']}): {result['error']}")
        else:
            print(f"\n✅ {result['feature']} ({result['window_type']}):")
            print(f"   Wavelet-Only:  MSE={result['mse_wavelet']:.6f}, Corr={result['corr_wavelet']:.4f}")
            print(f"   PCA+Wavelet:   MSE={result['mse_pca_wavelet']:.6f}, Corr={result['corr_pca_wavelet']:.4f}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())

