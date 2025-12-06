#!/usr/bin/env python3
"""
Script to create template scenarios for trained models.

Creates 3 template scenarios per model:
1. Baseline/Current Market: Recent past, unobserved future conditioning
2. Historical Stress Test: Simulation date from a major market event
3. Rising Rates/Inflation Scenario: Simulation date from a period of monetary tightening
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path
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
if api_url_raw.startswith("http://") and "run.app" in api_url_raw:
    API_URL = api_url_raw.replace("http://", "https://")
    logger.warning(f"⚠️  Forced HTTPS for API URL: {api_url_raw} -> {API_URL}")
else:
    API_URL = api_url_raw
logger.info(f"Using API URL: {API_URL}")

# Template scenarios for each specific model
# Scenarios are tailored to both conditioning set and target set
# Each scenario has at least one future conditioning window with a simulation date

SCENARIO_TEMPLATES_BY_MODEL = {
    # Balanced Portfolio Models
    "Balanced Portfolio - Core Macro Indicators": [
        {
            "name_suffix": "Economic Expansion - 2019",
            "description": "Simulates pre-COVID economic expansion period with stable growth conditions.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2019-05-15"
            },
            "portfolio_impact_theme": "Stable Growth",
            "assets_affected": [
                {"name": "Stocks", "direction": "up"},
                {"name": "Bonds", "direction": "flat"}
            ],
            "tags": ["expansion", "macro", "2019"]
        },
        {
            "name_suffix": "Post-COVID Recovery - 2021",
            "description": "Simulates post-COVID recovery with economic normalization.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2021-06-15"
            },
            "portfolio_impact_theme": "Recovery",
            "assets_affected": [
                {"name": "Stocks", "direction": "up"},
                {"name": "Bonds", "direction": "down"}
            ],
            "tags": ["recovery", "macro", "2021"]
        },
        {
            "name_suffix": "Moderate Growth - 2016",
            "description": "Simulates moderate economic growth period with steady expansion.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2016-08-15"
            },
            "portfolio_impact_theme": "Steady Growth",
            "assets_affected": [
                {"name": "Stocks", "direction": "up"},
                {"name": "Bonds", "direction": "flat"}
            ],
            "tags": ["growth", "macro", "2016"]
        },
    ],
    "Balanced Portfolio - Interest Rates & Yield Curve": [
        {
            "name_suffix": "Rising Rates Stress - 2022",
            "description": "Stress test with aggressive Fed rate hikes.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15"
            },
            "portfolio_impact_theme": "Rising Rates",
            "assets_affected": [
                {"name": "Bonds", "direction": "down"},
                {"name": "REITs", "direction": "down"},
                {"name": "Tech", "direction": "down"}
            ],
            "tags": ["stress-test", "rates", "fed-hikes", "2022"]
        },
        {
            "name_suffix": "Stable Rates Environment - 2018",
            "description": "Simulates stable rates environment with moderate growth.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2018-05-15"
            },
            "portfolio_impact_theme": "Stable Rates",
            "assets_affected": [
                {"name": "Bonds", "direction": "flat"},
                {"name": "Stocks", "direction": "up"}
            ],
            "tags": ["rates", "stable", "2018"]
        },
        {
            "name_suffix": "Low Rates Recovery - 2021",
            "description": "Simulates low rates recovery period post-COVID.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2021-06-15"
            },
            "portfolio_impact_theme": "Low Rates",
            "assets_affected": [
                {"name": "Bonds", "direction": "down"},
                {"name": "Stocks", "direction": "up"}
            ],
            "tags": ["rates", "recovery", "2021"]
        },
    ],
    "Balanced Portfolio - Core Macro Mix": [
        {
            "name_suffix": "Combined Stress - Rates + Volatility",
            "description": "Combines rising rates (2022) with high volatility (2020 COVID crash). Past windows use recent data. Future conditioning uses 'Interest Rates & Yield Curve' @ 2022-06-15 and 'VIX Volatility Index' @ 2020-03-15 to test balanced portfolio in dual stress.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15",
                "VIX Volatility Index": "2020-03-15"
            },
        },
        {
            "name_suffix": "Normal Bull Market - 2017",
            "description": "Simulates normal bull market with stable conditions. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2017-06-15 to test balanced portfolio in favorable market conditions.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2017-06-15"
            },
        },
        {
            "name_suffix": "Recovery with Low Volatility - 2021",
            "description": "Simulates post-COVID recovery with normalized volatility. Past windows use recent data. Future conditioning uses 'Core Macroeconomic Indicators' @ 2021-06-15 and 'VIX Volatility Index' @ 2021-06-15 to test balanced portfolio in recovery.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2021-06-15",
                "VIX Volatility Index": "2021-06-15"
            },
        },
    ],
    "Balanced Portfolio - Rates, Inflation & Currency Mix": [
        {
            "name_suffix": "Rising Rates + Inflation - 2022",
            "description": "Simulates 2022 environment with aggressive rate hikes and high inflation. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2022-06-15 to test balanced portfolio in inflationary environment.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15"
            },
        },
        {
            "name_suffix": "Combined Stress - Rates + Currency",
            "description": "Combines rising rates (2022) with currency volatility (2020 COVID). Past windows use recent data. Future conditioning uses 'Interest Rates & Yield Curve' @ 2022-06-15 and 'Currency Exchange Rates' @ 2020-03-15 to test balanced portfolio in dual stress.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15",
                "Currency Exchange Rates": "2020-03-15"
            },
        },
        {
            "name_suffix": "Stable Multi-Factor - 2018",
            "description": "Simulates stable environment across rates, inflation, and currency. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2018-05-15 to test balanced portfolio in stable multi-factor conditions.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2018-05-15"
            },
        },
    ],
    # Comprehensive Portfolio Models
    "Comprehensive Portfolio - Core Macro Indicators": [
        {
            "name_suffix": "Economic Expansion Multi-Asset - 2019",
            "description": "Simulates pre-COVID economic expansion period. Past windows use recent data. Future conditioning for 'Core Macroeconomic Indicators' uses 2019-05-15 to test comprehensive portfolio in stable growth across diverse assets.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2019-05-15"
            },
        },
        {
            "name_suffix": "Post-COVID Recovery Multi-Sector - 2021",
            "description": "Simulates post-COVID recovery with economic normalization. Past windows use recent data. Future conditioning for 'Core Macroeconomic Indicators' uses 2021-06-15 to test comprehensive portfolio performance during recovery across sectors.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2021-06-15"
            },
        },
        {
            "name_suffix": "Strong Growth Multi-Class - 2017",
            "description": "Simulates strong economic growth period. Past windows use recent data. Future conditioning for 'Core Macroeconomic Indicators' uses 2017-06-15 to test comprehensive portfolio in robust expansion across asset classes.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2017-06-15"
            },
        },
    ],
    "Comprehensive Portfolio - Interest Rates & Yield Curve": [
        {
            "name_suffix": "Rising Rates Stress Multi-Asset - 2022",
            "description": "Stress test with aggressive Fed rate hikes. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2022-06-15 to test comprehensive portfolio resilience to rising rates across diverse holdings.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15"
            },
        },
        {
            "name_suffix": "Stable Rates Multi-Sector - 2018",
            "description": "Simulates stable rates environment with moderate growth. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2018-05-15 to test comprehensive portfolio in stable rate conditions across sectors.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2018-05-15"
            },
        },
        {
            "name_suffix": "Low Rates Bull Market Multi-Class - 2017",
            "description": "Simulates low rates bull market period. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2017-06-15 to test comprehensive portfolio in accommodative monetary policy across asset classes.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2017-06-15"
            },
        },
    ],
    "Comprehensive Portfolio - Core Macro Mix": [
        {
            "name_suffix": "Combined Stress Multi-Asset - Rates + Volatility",
            "description": "Combines rising rates (2022) with high volatility (2020 COVID crash). Past windows use recent data. Future conditioning uses 'Interest Rates & Yield Curve' @ 2022-06-15 and 'VIX Volatility Index' @ 2020-03-15 to test comprehensive portfolio in dual stress across diverse assets.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15",
                "VIX Volatility Index": "2020-03-15"
            },
        },
        {
            "name_suffix": "Normal Bull Market Multi-Sector - 2017",
            "description": "Simulates normal bull market with stable conditions. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2017-06-15 to test comprehensive portfolio in favorable market conditions across sectors.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2017-06-15"
            },
        },
        {
            "name_suffix": "Recovery Normalized Volatility Multi-Class - 2021",
            "description": "Simulates post-COVID recovery with normalized volatility. Past windows use recent data. Future conditioning uses 'Core Macroeconomic Indicators' @ 2021-06-15 and 'VIX Volatility Index' @ 2021-06-15 to test comprehensive portfolio in recovery across asset classes.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Core Macroeconomic Indicators": "2021-06-15",
                "VIX Volatility Index": "2021-06-15"
            },
        },
    ],
    "Comprehensive Portfolio - Rates, Inflation & Currency Mix": [
        {
            "name_suffix": "Rising Rates + Inflation Multi-Asset - 2022",
            "description": "Simulates 2022 environment with aggressive rate hikes and high inflation. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2022-06-15 to test comprehensive portfolio in inflationary environment across diverse holdings.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15"
            },
        },
        {
            "name_suffix": "Combined Stress Multi-Sector - Rates + Currency",
            "description": "Combines rising rates (2022) with currency volatility (2020 COVID). Past windows use recent data. Future conditioning uses 'Interest Rates & Yield Curve' @ 2022-06-15 and 'Currency Exchange Rates' @ 2020-03-15 to test comprehensive portfolio in dual stress across sectors.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2022-06-15",
                "Currency Exchange Rates": "2020-03-15"
            },
        },
        {
            "name_suffix": "Stable Multi-Factor Multi-Class - 2018",
            "description": "Simulates stable environment across rates, inflation, and currency. Past windows use recent data. Future conditioning for 'Interest Rates & Yield Curve' uses 2018-05-15 to test comprehensive portfolio in stable multi-factor conditions across asset classes.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Interest Rates & Yield Curve": "2018-05-15"
            },
        },
    ],
}


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


async def get_models_for_project(db: CloudSQLManager, project_id: str) -> List[Dict[str, Any]]:
    """Get all models in the project"""
    async with db.connection() as conn:
        rows = await conn.fetch("""
            SELECT id, name, description, conditioning_set_id, target_set_id
            FROM models
            WHERE project_id = $1
            ORDER BY name
        """, db._to_uuid(project_id))
        
        return [dict(row) for row in rows]


async def get_feature_set_name(db: CloudSQLManager, feature_set_id: str) -> str:
    """Get feature set name by ID"""
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT name FROM feature_sets
            WHERE id = $1
        """, db._to_uuid(feature_set_id))
        
        if not row:
            return "Unknown"
        
        return row['name']


def get_scenario_templates_for_model(model_name: str) -> List[Dict[str, Any]]:
    """Get scenario templates for a model based on its exact model name"""
    # Return templates for this specific model, or default if not found
    templates = SCENARIO_TEMPLATES_BY_MODEL.get(model_name)
    
    if templates:
        return templates
    
    # Fallback for models not in our predefined list
    logger.warning(f"⚠️  No predefined scenarios for model '{model_name}', using defaults")
    return [
        {
            "name_suffix": "Normal Market - 2017",
            "description": "Simulates normal bull market conditions. Past windows use recent data. Future conditioning for primary feature group uses 2017-06-15.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Primary Feature Group": "2017-06-15"
            },
        },
        {
            "name_suffix": "Stress Test - 2020",
            "description": "Simulates COVID-19 market crash. Past windows use recent data. Future conditioning for primary feature group uses 2020-03-15.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Primary Feature Group": "2020-03-15"
            },
        },
        {
            "name_suffix": "Recovery - 2021",
            "description": "Simulates post-COVID recovery. Past windows use recent data. Future conditioning for primary feature group uses 2021-06-15.",
            "simulation_date": None,
            "feature_simulation_dates": {
                "Primary Feature Group": "2021-06-15"
            },
        },
    ]


def _serialize_for_json(obj: Any) -> Any:
    """Recursively convert UUID objects and other non-serializable types to strings"""
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool, type(None))):
        # Convert UUID and other objects to string
        try:
            return str(obj)
        except:
            return obj
    return obj


async def find_existing_template_scenario(
    db: CloudSQLManager,
    model_id: str,
    scenario_name: str,
) -> Optional[str]:
    """Find existing template scenario by name and model_id"""
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM scenarios
            WHERE model_id = $1
              AND name = $2
              AND is_template = TRUE
              AND status <> 'deleted'
            LIMIT 1
        """, db._to_uuid(model_id), scenario_name)
        
        if row:
            return str(row['id'])
        return None


async def create_or_update_template_scenario(
    api_url: str,
    api_key: str,
    db: CloudSQLManager,
    model_id: str,
    model_name: str,
    scenario_name: str,
    scenario_description: str,
    simulation_date: Optional[str] = None,
    feature_simulation_dates: Optional[Dict[str, Any]] = None,
    # New display fields
    portfolio_impact_theme: Optional[str] = None,
    assets_affected: Optional[List[Dict[str, str]]] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Create or update a template scenario (updates if exists, creates if not)"""
    # Convert model_id to string if it's a UUID object
    model_id_str = str(model_id) if model_id else None
    
    # Check if scenario already exists
    existing_scenario_id = await find_existing_template_scenario(
        db, model_id_str, scenario_name
    )
    
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    # Serialize all data for JSON (convert UUIDs to strings)
    payload_data = {
        "model_id": model_id_str,
        "name": scenario_name,
        "description": scenario_description,
        "simulation_date": simulation_date,
        "feature_simulation_dates": _serialize_for_json(feature_simulation_dates) if feature_simulation_dates else None,
        "is_template": True,
        # New display fields
        "portfolio_impact_theme": portfolio_impact_theme,
        "assets_affected": _serialize_for_json(assets_affected) if assets_affected else None,
        "tags": tags,
    }
    payload = _serialize_for_json(payload_data)
    
    if existing_scenario_id:
        # Update existing scenario using direct DB - bypasses API restrictions
        logger.info(f"📝 Updating existing template scenario: {scenario_name}")
        async with db.connection() as conn:
            await conn.execute("""
                UPDATE scenarios 
                SET description = $1,
                    simulation_date = $2,
                    feature_simulation_dates = $3,
                    portfolio_impact_theme = $4,
                    assets_affected = $5,
                    tags = $6,
                    status = 'configured',
                    updated_at = NOW()
                WHERE id = $7
            """,
                scenario_description,
                datetime.strptime(simulation_date, '%Y-%m-%d').date() if simulation_date else None,
                json.dumps(feature_simulation_dates) if feature_simulation_dates else None,
                portfolio_impact_theme,
                json.dumps(assets_affected) if assets_affected else None,
                json.dumps(tags) if tags else None,
                db._to_uuid(existing_scenario_id)
            )
        logger.info(f"✅ Updated template scenario: {existing_scenario_id}")
        return existing_scenario_id
    else:
        # Create new scenario using direct DB
        logger.info(f"Creating template scenario: {scenario_name}")
        async with db.connection() as conn:
            # Get model's owner for user_id
            model_row = await conn.fetchrow("""
                SELECT user_id, target_set_id, conditioning_set_id 
                FROM models WHERE id = $1
            """, db._to_uuid(model_id_str))
            
            if not model_row:
                raise Exception(f"Model {model_id_str} not found")
            
            user_id = model_row['user_id']
            target_set_id = model_row['target_set_id']
            conditioning_set_id = model_row['conditioning_set_id']
            
            row = await conn.fetchrow("""
                INSERT INTO scenarios (
                    user_id, model_id, target_set_id, conditioning_set_id, name, description, 
                    simulation_date, feature_simulation_dates, status, is_template, chat_history,
                    portfolio_impact_theme, assets_affected, tags
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'configured', TRUE, '[]'::jsonb, $9, $10, $11)
                RETURNING id
            """, 
                user_id, db._to_uuid(model_id_str), target_set_id, conditioning_set_id,
                scenario_name, scenario_description,
                datetime.strptime(simulation_date, '%Y-%m-%d').date() if simulation_date else None,
                json.dumps(feature_simulation_dates) if feature_simulation_dates else None,
                portfolio_impact_theme,
                json.dumps(assets_affected) if assets_affected else None,
                json.dumps(tags) if tags else None
            )
            scenario_id = str(row['id'])
            logger.info(f"✅ Created template scenario: {scenario_id}")
            return scenario_id


async def check_model_is_shared(db: CloudSQLManager, model_id: str) -> bool:
    """Check if model is shared (required for template scenarios)"""
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT is_shared FROM models
            WHERE id = $1
        """, db._to_uuid(model_id))
        
        if not row:
            return False
        
        return row.get('is_shared', False)


async def ensure_model_is_shared(api_url: str, api_key: str, model_id: str) -> bool:
    """Ensure model is shared (required for template scenarios)"""
    # First check if already shared
    db = CloudSQLManager()
    is_shared = await check_model_is_shared(db, model_id)
    
    if is_shared:
        logger.info(f"✅ Model {model_id} is already shared")
        return True
    
    # Try to share the model via API
    url = f"{api_url}/api/v1/models/{model_id}/share"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"is_shared": "true"}
    
    logger.info(f"Sharing model {model_id} (required for template scenarios)...")
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.patch(url, params=params, headers=headers)
            response.raise_for_status()
            logger.info(f"✅ Model {model_id} is now shared")
            return True
    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        logger.error(f"❌ Failed to share model {model_id}: {error_detail}")
        if "template" in error_detail.lower():
            logger.warning(f"⚠️  Model must be in a template project to be shared")
        elif "owner" in error_detail.lower() or "access" in error_detail.lower():
            logger.warning(f"⚠️  Only project owners can share models")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to share model {model_id}: {e}")
        logger.warning(f"⚠️  Template scenarios require shared models. Please share the model manually.")
        return False


async def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("Creating Template Scenarios for Models")
    logger.info("=" * 80)
    
    if not API_KEY:
        logger.warning("⚠️  API_KEY not set. Using unauthenticated requests (may fail)")
    
    db = CloudSQLManager()
    
    try:
        # Step 1: Get main project
        logger.info("\n📁 Step 1: Getting main project...")
        project_id = await get_template_project_id(db)
        logger.info(f"✅ Found project: {project_id}")
        
        # Step 2: Get all models
        logger.info("\n📊 Step 2: Getting models from project...")
        models = await get_models_for_project(db, project_id)
        logger.info(f"✅ Found {len(models)} models")
        
        if not models:
            logger.warning("⚠️  No models found. Please create models first using create_and_train_models.py")
            return
        
        # Step 3: Create template scenarios for each model
        logger.info(f"\n🎯 Step 3: Creating template scenarios for {len(models)} models...")
        logger.info("   Creating 3 template scenarios per model")
        
        created_scenarios = []
        skipped_models = []
        
        for i, model in enumerate(models, 1):
            # Convert UUID to string if needed
            model_id = str(model['id']) if model.get('id') else None
            model_name = model['name']
            
            logger.info(f"\n[{i}/{len(models)}] Processing model: {model_name}")
            logger.info(f"   Model ID: {model_id}")
            
            # Get conditioning set name for logging
            conditioning_set_id = model.get('conditioning_set_id')
            if conditioning_set_id:
                # Convert UUID to string if needed
                conditioning_set_id_str = str(conditioning_set_id) if conditioning_set_id else None
                conditioning_set_name = await get_feature_set_name(db, conditioning_set_id_str)
                logger.info(f"   Conditioning Set: {conditioning_set_name}")
            else:
                conditioning_set_name = "Unknown"
            
            # Ensure model is shared (required for template scenarios)
            is_shared = await ensure_model_is_shared(API_URL, API_KEY, model_id)
            if not is_shared:
                logger.warning(f"⚠️  Skipping model {model_name} - not shared (required for template scenarios)")
                skipped_models.append(model_name)
                continue
            
            # Get appropriate scenario templates for this specific model
            scenario_templates = get_scenario_templates_for_model(model_name)
            
            # Create or update scenarios for this model
            for j, template in enumerate(scenario_templates, 1):
                # Use just the name_suffix, not the full model name
                scenario_name = template['name_suffix']
                
                try:
                    scenario_id = await create_or_update_template_scenario(
                        API_URL,
                        API_KEY,
                        db,
                        model_id,
                        model_name,
                        scenario_name,
                        template['description'],
                        template.get('simulation_date'),
                        template.get('feature_simulation_dates'),
                        # New display fields
                        template.get('portfolio_impact_theme'),
                        template.get('assets_affected'),
                        template.get('tags'),
                    )
                    
                    created_scenarios.append({
                        "model_name": model_name,
                        "model_id": model_id,
                        "scenario_name": scenario_name,
                        "scenario_id": scenario_id,
                        "simulation_date": template.get('simulation_date'),
                    })
                    
                    logger.info(f"✅ Processed scenario {j}/{len(scenario_templates)}: {scenario_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create scenario '{scenario_name}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ OPERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nProject ID: {project_id}")
        logger.info(f"Processed: {len(models)} models")
        logger.info(f"Created: {len(created_scenarios)} template scenarios")
        logger.info(f"Skipped: {len(skipped_models)} models (not shared)")
        logger.info(f"Total: {len(created_scenarios)}/{len(models) * 3} scenarios\n")
        
        if created_scenarios:
            logger.info("Created template scenarios:")
            current_model = None
            for idx, scenario_info in enumerate(created_scenarios, 1):
                if scenario_info['model_name'] != current_model:
                    current_model = scenario_info['model_name']
                    logger.info(f"\n  Model: {current_model}")
                logger.info(f"    {idx}. {scenario_info['scenario_name']}")
                logger.info(f"       - ID: {scenario_info['scenario_id']}")
                if scenario_info['simulation_date']:
                    logger.info(f"       - Simulation Date: {scenario_info['simulation_date']}")
                else:
                    logger.info(f"       - Simulation Date: Recent (baseline)")
        
        if skipped_models:
            logger.info(f"\n⚠️  Skipped models (not shared):")
            for model_name in skipped_models:
                logger.info(f"  - {model_name}")
            logger.info("\n💡 Tip: Share models to enable template scenarios")
        
        logger.info("\n📝 Next Steps:")
        logger.info("  1. Template scenarios are now visible to all users for shared models")
        logger.info("  2. Users can clone template scenarios to create their own copies")
        logger.info("  3. Users can run simulations with their own parameters (n_paths, etc.)")
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

