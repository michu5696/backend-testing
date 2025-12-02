#!/usr/bin/env python3
"""
Script to create template target sets for portfolio management
Creates 10 diverse target sets optimized for options trading and portfolio construction
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import date

# Add backend directory to path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_dir)

# Load environment variables from backend/.env
from dotenv import load_dotenv
env_path = Path(backend_dir) / '.env'
load_dotenv(dotenv_path=env_path)

from cloudsql_client import CloudSQLManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the 10 target sets with their assets
# Mix of individual stocks, ETFs, bonds, commodities, and REITs
# All assets verified to be available on Yahoo Finance
TARGET_SETS = [
    {
        "name": "US Mega Cap Tech Stocks",
        "description": "Large-cap technology companies (individual stocks). Highly liquid with active options markets. Ideal for tech exposure and options strategies.",
        "assets": [
            {"id": "AAPL", "name": "Apple Inc"},
            {"id": "MSFT", "name": "Microsoft Corporation"},
            {"id": "GOOGL", "name": "Alphabet Inc"},
            {"id": "AMZN", "name": "Amazon.com Inc"},
            {"id": "META", "name": "Meta Platforms Inc"},
            {"id": "NVDA", "name": "NVIDIA Corporation"},
            {"id": "TSLA", "name": "Tesla Inc"},
            {"id": "NFLX", "name": "Netflix Inc"},
        ]
    },
    {
        "name": "Major Index ETFs",
        "description": "Highly liquid index ETFs with the most active options markets globally. Perfect for market exposure and hedging.",
        "assets": [
            {"id": "SPY", "name": "SPDR S&P 500 ETF"},
            {"id": "QQQ", "name": "Invesco QQQ Trust"},
            {"id": "IWM", "name": "iShares Russell 2000 ETF"},
            {"id": "DIA", "name": "SPDR Dow Jones Industrial Average ETF"},
            {"id": "VTI", "name": "Vanguard Total Stock Market ETF"},
            {"id": "EFA", "name": "iShares MSCI EAFE ETF"},
        ]
    },
    {
        "name": "US Financial Stocks",
        "description": "Major financial institutions and payment networks (individual stocks). Interest rate sensitive with active options trading.",
        "assets": [
            {"id": "JPM", "name": "JPMorgan Chase & Co"},
            {"id": "BAC", "name": "Bank of America Corp"},
            {"id": "GS", "name": "Goldman Sachs Group Inc"},
            {"id": "WFC", "name": "Wells Fargo & Company"},
            {"id": "C", "name": "Citigroup Inc"},
            {"id": "MS", "name": "Morgan Stanley"},
            {"id": "V", "name": "Visa Inc"},
            {"id": "MA", "name": "Mastercard Inc"},
        ]
    },
    {
        "name": "Healthcare & Pharma Stocks",
        "description": "Major healthcare and pharmaceutical companies (individual stocks). Defensive sector with growth potential.",
        "assets": [
            {"id": "JNJ", "name": "Johnson & Johnson"},
            {"id": "UNH", "name": "UnitedHealth Group Inc"},
            {"id": "PFE", "name": "Pfizer Inc"},
            {"id": "ABBV", "name": "AbbVie Inc"},
            {"id": "MRK", "name": "Merck & Co Inc"},
            {"id": "TMO", "name": "Thermo Fisher Scientific Inc"},
            {"id": "ABT", "name": "Abbott Laboratories"},
            {"id": "DHR", "name": "Danaher Corporation"},
        ]
    },
    {
        "name": "Energy Stocks & Commodities",
        "description": "Energy companies (individual stocks) and commodity ETFs. Inflation hedge and commodity exposure.",
        "assets": [
            {"id": "XOM", "name": "Exxon Mobil Corporation"},
            {"id": "CVX", "name": "Chevron Corporation"},
            {"id": "COP", "name": "ConocoPhillips"},
            {"id": "SLB", "name": "Schlumberger Limited"},
            {"id": "GLD", "name": "SPDR Gold Trust"},
            {"id": "SLV", "name": "iShares Silver Trust"},
            {"id": "USO", "name": "United States Oil Fund"},
            {"id": "UNG", "name": "United States Natural Gas Fund"},
        ]
    },
    {
        "name": "Fixed Income & Treasury ETFs",
        "description": "Treasury and bond ETFs across duration spectrum. Used for fixed income allocation and rates trading.",
        "assets": [
            {"id": "TLT", "name": "iShares 20+ Year Treasury Bond ETF"},
            {"id": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF"},
            {"id": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF"},
            {"id": "AGG", "name": "iShares Core U.S. Aggregate Bond ETF"},
            {"id": "LQD", "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF"},
            {"id": "HYG", "name": "iShares iBoxx $ High Yield Corporate Bond ETF"},
            {"id": "TIP", "name": "iShares TIPS Bond ETF"},
            {"id": "MUB", "name": "iShares National Muni Bond ETF"},
        ]
    },
    {
        "name": "Consumer & Retail Stocks",
        "description": "Major consumer discretionary and retail companies (individual stocks). Economic cycle sensitive.",
        "assets": [
            {"id": "WMT", "name": "Walmart Inc"},
            {"id": "HD", "name": "The Home Depot Inc"},
            {"id": "MCD", "name": "McDonald's Corporation"},
            {"id": "NKE", "name": "Nike Inc"},
            {"id": "SBUX", "name": "Starbucks Corporation"},
            {"id": "TGT", "name": "Target Corporation"},
            {"id": "LOW", "name": "Lowe's Companies Inc"},
            {"id": "COST", "name": "Costco Wholesale Corporation"},
        ]
    },
    {
        "name": "Real Estate REITs",
        "description": "Real Estate Investment Trusts (individual REITs) and real estate ETFs. Income-focused with real estate exposure.",
        "assets": [
            {"id": "AMT", "name": "American Tower Corporation"},
            {"id": "PLD", "name": "Prologis Inc"},
            {"id": "EQIX", "name": "Equinix Inc"},
            {"id": "PSA", "name": "Public Storage"},
            {"id": "WELL", "name": "Welltower Inc"},
            {"id": "IYR", "name": "iShares U.S. Real Estate ETF"},
            {"id": "VNQ", "name": "Vanguard Real Estate ETF"},
            {"id": "SCHH", "name": "Schwab U.S. REIT ETF"},
        ]
    },
    {
        "name": "Industrial & Materials Stocks",
        "description": "Industrial and materials companies (individual stocks). Economic growth indicators and infrastructure plays.",
        "assets": [
            {"id": "BA", "name": "The Boeing Company"},
            {"id": "CAT", "name": "Caterpillar Inc"},
            {"id": "GE", "name": "General Electric Company"},
            {"id": "HON", "name": "Honeywell International Inc"},
            {"id": "LIN", "name": "Linde plc"},
            {"id": "APD", "name": "Air Products and Chemicals Inc"},
            {"id": "ECL", "name": "Ecolab Inc"},
            {"id": "EMR", "name": "Emerson Electric Co"},
        ]
    },
    {
        "name": "International Stocks & ETFs",
        "description": "International developed and emerging market stocks and ETFs. Geographic diversification.",
        "assets": [
            {"id": "ASML", "name": "ASML Holding NV"},
            {"id": "TSM", "name": "Taiwan Semiconductor Manufacturing"},
            {"id": "NVO", "name": "Novo Nordisk A/S"},
            {"id": "SAP", "name": "SAP SE"},
            {"id": "EEM", "name": "iShares MSCI Emerging Markets ETF"},
            {"id": "EWJ", "name": "iShares MSCI Japan ETF"},
            {"id": "EWZ", "name": "iShares MSCI Brazil ETF"},
            {"id": "FXI", "name": "iShares China Large-Cap ETF"},
        ]
    },
]


async def get_or_create_template_project(db: CloudSQLManager) -> str:
    """Get existing template project or create one"""
    
    # Try to find existing template project
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM projects 
            WHERE is_template = true 
            AND name = 'Portfolio Templates'
            LIMIT 1
        """)
        
        if row:
            project_id = str(row['id'])
            logger.info(f"✅ Found existing template project: {project_id}")
            return project_id
    
    # Create new template project
    logger.info("Creating new template project...")
    
    # Need an admin user ID - let's get the first admin
    async with db.connection() as conn:
        admin_row = await conn.fetchrow("""
            SELECT id FROM users 
            WHERE is_admin = true 
            LIMIT 1
        """)
        
        if not admin_row:
            raise Exception("No admin user found. Please create an admin user first.")
        
        admin_id = str(admin_row['id'])
    
    # Create project
    async with db.connection() as conn:
        project_id = await conn.fetchval("""
            INSERT INTO projects (
                user_id,
                name,
                description,
                is_template,
                training_start_date,
                training_end_date
            ) VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """,
            db._to_uuid(admin_id),
            "Portfolio Templates",
            "Template project containing pre-configured target sets for portfolio management and options trading",
            True,
            date(2020, 1, 1),  # Default training window
            date(2024, 12, 31)
        )
        
        project_id = str(project_id)
        logger.info(f"✅ Created template project: {project_id}")
        return project_id


async def create_target_set(
    db: CloudSQLManager, 
    project_id: str, 
    set_data: dict
) -> str:
    """Create a single target feature set"""
    
    name = set_data["name"]
    description = set_data["description"]
    assets = set_data["assets"]
    
    # Prepare features array
    features = [
        {
            "id": asset["id"],
            "source": "Yahoo",
            "name": asset["name"]
        }
        for asset in assets
    ]
    
    # Prepare data collectors
    data_collectors = [
        {
            "source": "Yahoo",
            "api_key": None
        }
    ]
    
    # Create feature set
    feature_set_id = await db.create_feature_set(
        project_id=project_id,
        name=name,
        description=description,
        set_type="target"
    )
    
    # Update with features and data collectors
    async with db.connection() as conn:
        await conn.execute("""
            UPDATE feature_sets
            SET features = $1::jsonb,
                data_collectors = $2::jsonb
            WHERE id = $3
        """,
            json.dumps(features),
            json.dumps(data_collectors),
            db._to_uuid(feature_set_id)
        )
    
    logger.info(f"✅ Created target set '{name}' with {len(assets)} assets (ID: {feature_set_id})")
    return feature_set_id


async def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("Creating Template Target Sets for Portfolio Management")
    logger.info("=" * 80)
    
    db = CloudSQLManager()
    
    try:
        # Step 1: Get or create template project
        logger.info("\n📁 Step 1: Setting up template project...")
        project_id = await get_or_create_template_project(db)
        
        # Step 2: Create/update each target set
        logger.info(f"\n🎯 Step 2: Creating/updating {len(TARGET_SETS)} target sets...")
        
        created_sets = []
        updated_sets = []
        for i, set_data in enumerate(TARGET_SETS, 1):
            logger.info(f"\n[{i}/{len(TARGET_SETS)}] Processing: {set_data['name']}")
            try:
                # Check if already exists
                async with db.connection() as conn:
                    existing = await conn.fetchrow("""
                        SELECT id, features, data_collectors FROM feature_sets
                        WHERE project_id = $1 
                        AND name = $2
                        AND set_type = 'target'
                    """, db._to_uuid(project_id), set_data['name'])
                
                if existing:
                    # Update existing set with proper features/data_collectors
                    logger.info(f"🔄 Updating existing target set '{set_data['name']}'")
                    
                    features = [
                        {
                            "id": asset["id"],
                            "source": "Yahoo",
                            "name": asset["name"]
                        }
                        for asset in set_data["assets"]
                    ]
                    
                    data_collectors = [
                        {
                            "source": "Yahoo",
                            "api_key": None
                        }
                    ]
                    
                    async with db.connection() as conn:
                        await conn.execute("""
                            UPDATE feature_sets
                            SET features = $1::jsonb,
                                data_collectors = $2::jsonb,
                                description = $3
                            WHERE id = $4
                        """,
                            json.dumps(features),
                            json.dumps(data_collectors),
                            set_data["description"],
                            existing['id']
                        )
                    
                    updated_sets.append({
                        "name": set_data["name"],
                        "id": str(existing['id']),
                        "assets_count": len(set_data["assets"])
                    })
                    logger.info(f"✅ Updated '{set_data['name']}' with {len(set_data['assets'])} assets")
                else:
                    # Create new set
                    feature_set_id = await create_target_set(db, project_id, set_data)
                    created_sets.append({
                        "name": set_data["name"],
                        "id": feature_set_id,
                        "assets_count": len(set_data["assets"])
                    })
            except Exception as e:
                logger.error(f"❌ Failed to process '{set_data['name']}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ TARGET SETS OPERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nProject ID: {project_id}")
        logger.info(f"Created: {len(created_sets)} new target sets")
        logger.info(f"Updated: {len(updated_sets)} existing target sets")
        logger.info(f"Total: {len(created_sets) + len(updated_sets)}/{len(TARGET_SETS)}\n")
        
        if created_sets:
            logger.info("Newly created target sets:")
            for idx, set_info in enumerate(created_sets, 1):
                logger.info(f"  {idx}. {set_info['name']}")
                logger.info(f"     - ID: {set_info['id']}")
                logger.info(f"     - Assets: {set_info['assets_count']}")
        
        if updated_sets:
            logger.info("\nUpdated target sets:")
            for idx, set_info in enumerate(updated_sets, 1):
                logger.info(f"  {idx}. {set_info['name']}")
                logger.info(f"     - ID: {set_info['id']}")
                logger.info(f"     - Assets: {set_info['assets_count']}")
        
        logger.info("\n📝 Next Steps:")
        logger.info("  1. Users can now create portfolios using these target sets")
        logger.info("  2. Train models for each target set to enable portfolio simulation")
        logger.info("  3. Fetch data for target sets using: POST /feature-sets/{id}/fetch-data")
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

