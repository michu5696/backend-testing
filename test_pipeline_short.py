#!/usr/bin/env python3
"""
Short test pipeline for fast local iteration (2020-2025, fewer samples).

Usage:
    python test_pipeline_short.py
"""

import sys
import os

# Import the main workflow
from test_pipeline import main, DEFAULT_CONDITIONING_FEATURES, DEFAULT_TARGET_FEATURES

if __name__ == "__main__":
    # Override with shorter date range and local backend
    # Start with defaults
    args = [
        "--api-url", "http://localhost:8000",
        "--project-name", "Treasury ETFs (Short Test)",
        "--project-description", "Fast iteration test: 2020-2025",
        "--training-start", "2020-01-01",
        "--training-end", "2025-01-01",
        "--past-window", "100",
        "--future-window", "80",
        "--stride", "2",  # Stride = 2 for more samples
        "--training-split", "0.8",
        "--n-regimes", "3",
        "--scenario-date", "2024-01-15",
        "--scenario-name", "2024 Test",
        "--model-name", "Treasury ETF Model (Short)",
        "--model-description", "Fast iteration model with shorter date range",
        "--training-timeout", "0",  # No timeout - wait as long as needed
        "--poll-interval", "5",
        "--wavelet-past", "db2",  # Coarser decomposition for past
        "--wavelet-future", "db4",  # Better reconstruction for future
    ]
    
    # Check if user passed --stage argument
    if "--stage" not in sys.argv:
        # Default to training if not specified
        args.extend(["--stage", "training"])
    else:
        # Use the stage from command line
        stage_idx = sys.argv.index("--stage")
        if stage_idx + 1 < len(sys.argv):
            args.extend(["--stage", sys.argv[stage_idx + 1]])
    
    # Check if user passed --resume-from argument
    if "--resume-from" in sys.argv:
        resume_idx = sys.argv.index("--resume-from")
        if resume_idx + 1 < len(sys.argv):
            args.extend(["--resume-from", sys.argv[resume_idx + 1]])
    
    # Add API key from environment or file
    api_key = os.getenv("SABLIER_API_KEY")
    if not api_key:
        try:
            with open("../backend/admin/admin_api_key_v2.txt", "r") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print("ERROR: No API key found. Set SABLIER_API_KEY or create admin_api_key_v2.txt", file=sys.stderr)
            sys.exit(1)
    
    args.extend(["--api-key", api_key])
    
    # Add FRED API key
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        args.extend(["--fred-api-key", fred_key])
    
    print("=" * 80)
    print("🚀 SHORT TEST PIPELINE (2020-2025, Local Backend)")
    print("=" * 80)
    print(f"Backend: http://localhost:8000")
    print(f"Date range: 2020-01-01 to 2025-01-01")
    print(f"Stride: 2 (more samples for better training)")
    print(f"Conditioning features: {len(DEFAULT_CONDITIONING_FEATURES)}")
    print(f"Target features: {len(DEFAULT_TARGET_FEATURES)}")
    print("=" * 80)
    print()
    
    main(args)

