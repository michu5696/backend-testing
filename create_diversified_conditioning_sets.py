#!/usr/bin/env python3
"""
Script to create template conditioning sets for scenario configuration
Creates diverse conditioning sets with macroeconomic and market indicators
from FRED and Yahoo Finance for simulating scenarios across target sets
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


# Define diversified conditioning sets with mixed factor types
# Each set contains a mix of rates, inflation, labor, volatility, currency, commodities, credit, etc.
# This enables scenario configuration with diverse conditioning factors, not just one type
CONDITIONING_SETS = [
    {
        "name": "Core Macroeconomic Mix",
        "description": "Diversified mix of key macro factors: interest rates, inflation, GDP, unemployment, and volatility. Enables scenario configuration across multiple macro dimensions.",
        "features": [
            # Interest Rates (FRED)
            {"id": "FEDFUNDS", "name": "Federal Funds Effective Rate", "source": "FRED"},
            {"id": "DGS10", "name": "10-Year Treasury Constant Maturity Rate", "source": "FRED"},
            {"id": "DGS2", "name": "2-Year Treasury Constant Maturity Rate", "source": "FRED"},
            # Inflation (FRED)
            {"id": "CPIAUCSL", "name": "Consumer Price Index for All Urban Consumers: All Items", "source": "FRED"},
            {"id": "CPILFESL", "name": "Consumer Price Index for All Urban Consumers: All Items Less Food & Energy", "source": "FRED"},
            # Economic Activity (FRED)
            {"id": "GDPC1", "name": "Real Gross Domestic Product", "source": "FRED"},
            {"id": "UNRATE", "name": "Unemployment Rate", "source": "FRED"},
            # Market Volatility (Yahoo)
            {"id": "^VIX", "name": "CBOE Volatility Index (VIX)", "source": "Yahoo"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None},
            {"source": "Yahoo", "api_key": None}
        ]
    },
    {
        "name": "Rates, Inflation & Currency Mix",
        "description": "Mix of interest rates, inflation indicators, and currency factors. Enables scenario configuration across monetary policy, price levels, and FX dimensions.",
        "features": [
            # Interest Rates (FRED)
            {"id": "FEDFUNDS", "name": "Federal Funds Effective Rate", "source": "FRED"},
            {"id": "DGS10", "name": "10-Year Treasury Constant Maturity Rate", "source": "FRED"},
            {"id": "DGS2", "name": "2-Year Treasury Constant Maturity Rate", "source": "FRED"},
            {"id": "T10Y2Y", "name": "10-Year Treasury Minus 2-Year Treasury Constant Maturity", "source": "FRED"},
            # Inflation (FRED)
            {"id": "CPIAUCSL", "name": "Consumer Price Index for All Urban Consumers: All Items", "source": "FRED"},
            {"id": "CPILFESL", "name": "Consumer Price Index for All Urban Consumers: All Items Less Food & Energy", "source": "FRED"},
            {"id": "PPIACO", "name": "Producer Price Index for All Commodities", "source": "FRED"},
            # Currency (Yahoo)
            {"id": "DX-Y.NYB", "name": "US Dollar Index", "source": "Yahoo"},
            {"id": "EURUSD=X", "name": "EUR/USD Exchange Rate", "source": "Yahoo"},
            {"id": "JPY=X", "name": "USD/JPY Exchange Rate", "source": "Yahoo"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None},
            {"source": "Yahoo", "api_key": None}
        ]
    },
    {
        "name": "Labor, Growth & Credit Mix",
        "description": "Mix of labor market indicators, economic growth metrics, and credit spreads. Enables scenario configuration across employment, growth, and credit risk dimensions.",
        "features": [
            # Labor Market (FRED)
            {"id": "UNRATE", "name": "Unemployment Rate", "source": "FRED"},
            {"id": "PAYEMS", "name": "All Employees, Total Nonfarm", "source": "FRED"},
            {"id": "CES0500000003", "name": "Average Hourly Earnings of All Employees, Total Private", "source": "FRED"},
            {"id": "JTSJOL", "name": "Job Openings: Total Nonfarm", "source": "FRED"},
            # Economic Growth (FRED)
            {"id": "GDPC1", "name": "Real Gross Domestic Product", "source": "FRED"},
            # Credit Spreads (FRED)
            {"id": "BAMLH0A0HYM2", "name": "ICE BofA US High Yield Index Option-Adjusted Spread", "source": "FRED"},
            {"id": "BAMLC0A0CM", "name": "ICE BofA US Corporate Index Option-Adjusted Spread", "source": "FRED"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None}
        ]
    },
    {
        "name": "Volatility, Currency & Commodities Mix",
        "description": "Mix of market volatility, currency pairs, and commodity prices. Enables scenario configuration across risk, FX, and commodity dimensions.",
        "features": [
            # Market Volatility (Yahoo)
            {"id": "^VIX", "name": "CBOE Volatility Index (VIX)", "source": "Yahoo"},
            {"id": "^VXN", "name": "CBOE Nasdaq Volatility Index", "source": "Yahoo"},
            # Currency (Yahoo)
            {"id": "DX-Y.NYB", "name": "US Dollar Index", "source": "Yahoo"},
            {"id": "EURUSD=X", "name": "EUR/USD Exchange Rate", "source": "Yahoo"},
            {"id": "GBPUSD=X", "name": "GBP/USD Exchange Rate", "source": "Yahoo"},
            {"id": "JPY=X", "name": "USD/JPY Exchange Rate", "source": "Yahoo"},
            # Commodities (Yahoo)
            {"id": "CL=F", "name": "Crude Oil WTI Futures", "source": "Yahoo"},
            {"id": "GC=F", "name": "Gold Futures", "source": "Yahoo"},
            {"id": "SI=F", "name": "Silver Futures", "source": "Yahoo"},
        ],
        "data_collectors": [
            {"source": "Yahoo", "api_key": None}
        ]
    },
    {
        "name": "Monetary Policy & Inflation Mix",
        "description": "Mix of interest rates, money supply, and inflation indicators. Enables scenario configuration across monetary policy and price level dimensions.",
        "features": [
            # Interest Rates (FRED)
            {"id": "FEDFUNDS", "name": "Federal Funds Effective Rate", "source": "FRED"},
            {"id": "DGS10", "name": "10-Year Treasury Constant Maturity Rate", "source": "FRED"},
            {"id": "DGS2", "name": "2-Year Treasury Constant Maturity Rate", "source": "FRED"},
            # Money Supply (FRED)
            {"id": "M2SL", "name": "M2 Money Stock", "source": "FRED"},
            {"id": "M2V", "name": "Velocity of M2 Money Stock", "source": "FRED"},
            # Inflation (FRED)
            {"id": "CPIAUCSL", "name": "Consumer Price Index for All Urban Consumers: All Items", "source": "FRED"},
            {"id": "CPILFESL", "name": "Consumer Price Index for All Urban Consumers: All Items Less Food & Energy", "source": "FRED"},
            {"id": "PCEPI", "name": "Personal Consumption Expenditures: Chain-type Price Index", "source": "FRED"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None}
        ]
    },
    {
        "name": "Credit, Housing & Real Estate Mix",
        "description": "Mix of credit spreads, housing indicators, and real estate metrics. Enables scenario configuration across credit risk, housing, and real estate dimensions.",
        "features": [
            # Credit Spreads (FRED)
            {"id": "BAMLH0A0HYM2", "name": "ICE BofA US High Yield Index Option-Adjusted Spread", "source": "FRED"},
            {"id": "BAMLC0A0CM", "name": "ICE BofA US Corporate Index Option-Adjusted Spread", "source": "FRED"},
            {"id": "BAMLC0A4CBBB", "name": "ICE BofA BBB US Corporate Index Option-Adjusted Spread", "source": "FRED"},
            # Housing (FRED)
            {"id": "HOUST", "name": "Housing Starts: Total New Privately Owned", "source": "FRED"},
            {"id": "PERMIT", "name": "New Private Housing Units Authorized by Building Permits", "source": "FRED"},
            {"id": "MSPUS", "name": "Median Sales Price of Houses Sold for the United States", "source": "FRED"},
            {"id": "USSTHPI", "name": "All-Transactions House Price Index for the United States", "source": "FRED"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None}
        ]
    },
    {
        "name": "Commodities, Inflation & Currency Mix",
        "description": "Mix of commodity prices, inflation indicators, and currency factors. Enables scenario configuration across commodity-driven, inflation, and FX dimensions.",
        "features": [
            # Commodities (Yahoo)
            {"id": "CL=F", "name": "Crude Oil WTI Futures", "source": "Yahoo"},
            {"id": "BZ=F", "name": "Brent Crude Oil Futures", "source": "Yahoo"},
            {"id": "GC=F", "name": "Gold Futures", "source": "Yahoo"},
            {"id": "SI=F", "name": "Silver Futures", "source": "Yahoo"},
            {"id": "NG=F", "name": "Natural Gas Futures", "source": "Yahoo"},
            # Inflation (FRED)
            {"id": "CPIAUCSL", "name": "Consumer Price Index for All Urban Consumers: All Items", "source": "FRED"},
            {"id": "PPIACO", "name": "Producer Price Index for All Commodities", "source": "FRED"},
            # Currency (Yahoo)
            {"id": "DX-Y.NYB", "name": "US Dollar Index", "source": "Yahoo"},
            {"id": "EURUSD=X", "name": "EUR/USD Exchange Rate", "source": "Yahoo"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None},
            {"source": "Yahoo", "api_key": None}
        ]
    },
    {
        "name": "Risk & Market Sentiment Mix",
        "description": "Mix of volatility indices, credit spreads, and currency factors. Enables scenario configuration across risk, sentiment, and market stress dimensions.",
        "features": [
            # Volatility (Yahoo)
            {"id": "^VIX", "name": "CBOE Volatility Index (VIX)", "source": "Yahoo"},
            {"id": "^VXN", "name": "CBOE Nasdaq Volatility Index", "source": "Yahoo"},
            {"id": "^GVZ", "name": "CBOE Gold Volatility Index", "source": "Yahoo"},
            # Credit Spreads (FRED)
            {"id": "BAMLH0A0HYM2", "name": "ICE BofA US High Yield Index Option-Adjusted Spread", "source": "FRED"},
            {"id": "BAMLC0A0CM", "name": "ICE BofA US Corporate Index Option-Adjusted Spread", "source": "FRED"},
            # Currency (Yahoo)
            {"id": "DX-Y.NYB", "name": "US Dollar Index", "source": "Yahoo"},
            {"id": "EURUSD=X", "name": "EUR/USD Exchange Rate", "source": "Yahoo"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None},
            {"source": "Yahoo", "api_key": None}
        ]
    },
    {
        "name": "Economic Activity & Labor Mix",
        "description": "Mix of GDP, employment, wages, and productivity indicators. Enables scenario configuration across economic growth and labor market dimensions.",
        "features": [
            # Economic Growth (FRED)
            {"id": "GDPC1", "name": "Real Gross Domestic Product", "source": "FRED"},
            # Labor Market (FRED)
            {"id": "UNRATE", "name": "Unemployment Rate", "source": "FRED"},
            {"id": "PAYEMS", "name": "All Employees, Total Nonfarm", "source": "FRED"},
            {"id": "U6RATE", "name": "Total Unemployed, Plus All Persons Marginally Attached to the Labor Force", "source": "FRED"},
            {"id": "CES0500000003", "name": "Average Hourly Earnings of All Employees, Total Private", "source": "FRED"},
            {"id": "JTSJOL", "name": "Job Openings: Total Nonfarm", "source": "FRED"},
            {"id": "ICSA", "name": "Initial Claims", "source": "FRED"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None}
        ]
    },
    {
        "name": "Comprehensive Macro & Market Mix",
        "description": "Maximum diversification across all factor types: rates, inflation, labor, volatility, currency, commodities, credit, housing, and money supply. Enables full scenario configuration across all major conditioning dimensions.",
        "features": [
            # Key Rates (FRED)
            {"id": "FEDFUNDS", "name": "Federal Funds Effective Rate", "source": "FRED"},
            {"id": "DGS10", "name": "10-Year Treasury Constant Maturity Rate", "source": "FRED"},
            {"id": "DGS2", "name": "2-Year Treasury Constant Maturity Rate", "source": "FRED"},
            # Key Inflation (FRED)
            {"id": "CPIAUCSL", "name": "Consumer Price Index for All Urban Consumers: All Items", "source": "FRED"},
            {"id": "CPILFESL", "name": "Consumer Price Index for All Urban Consumers: All Items Less Food & Energy", "source": "FRED"},
            # Key Economic Activity (FRED)
            {"id": "GDPC1", "name": "Real Gross Domestic Product", "source": "FRED"},
            {"id": "UNRATE", "name": "Unemployment Rate", "source": "FRED"},
            # Market Volatility (Yahoo)
            {"id": "^VIX", "name": "CBOE Volatility Index (VIX)", "source": "Yahoo"},
            # Currency (Yahoo)
            {"id": "DX-Y.NYB", "name": "US Dollar Index", "source": "Yahoo"},
            {"id": "EURUSD=X", "name": "EUR/USD Exchange Rate", "source": "Yahoo"},
            # Commodities (Yahoo)
            {"id": "CL=F", "name": "Crude Oil WTI Futures", "source": "Yahoo"},
            {"id": "GC=F", "name": "Gold Futures", "source": "Yahoo"},
            # Credit Spreads (FRED)
            {"id": "BAMLH0A0HYM2", "name": "ICE BofA US High Yield Index Option-Adjusted Spread", "source": "FRED"},
        ],
        "data_collectors": [
            {"source": "FRED", "api_key": None},
            {"source": "Yahoo", "api_key": None}
        ]
    },
]


async def get_or_create_template_project(db: CloudSQLManager) -> str:
    """Get existing main project or create one"""
    
    # Try to find existing main project
    async with db.connection() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM projects 
            WHERE is_template = true 
            AND name = 'main project'
            LIMIT 1
        """)
        
        if row:
            project_id = str(row['id'])
            logger.info(f"✅ Found existing main project: {project_id}")
            return project_id
    
    # Create new main project
    logger.info("Creating new main project...")
    
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
            "main project",
            "Main project containing pre-configured target sets and conditioning sets for portfolio management and options trading",
            True,
            date(2020, 1, 1),  # Default training window
            date(2024, 12, 31)
        )
        
        project_id = str(project_id)
        logger.info(f"✅ Created main project: {project_id}")
        return project_id


async def create_conditioning_set(
    db: CloudSQLManager, 
    project_id: str, 
    set_data: dict
) -> str:
    """Create a single conditioning feature set"""
    
    name = set_data["name"]
    description = set_data["description"]
    features = set_data["features"]
    data_collectors = set_data["data_collectors"]
    
    # Prepare features array (already in correct format)
    features_array = [
        {
            "id": feature["id"],
            "source": feature["source"],
            "name": feature["name"]
        }
        for feature in features
    ]
    
    # Prepare data collectors array
    collectors_array = [
        {
            "source": collector["source"],
            "api_key": collector.get("api_key")
        }
        for collector in data_collectors
    ]
    
    # Create feature set
    feature_set_id = await db.create_feature_set(
        project_id=project_id,
        name=name,
        description=description,
        set_type="conditioning"
    )
    
    # Update with features and data collectors
    async with db.connection() as conn:
        await conn.execute("""
            UPDATE feature_sets
            SET features = $1::jsonb,
                data_collectors = $2::jsonb
            WHERE id = $3
        """,
            json.dumps(features_array),
            json.dumps(collectors_array),
            db._to_uuid(feature_set_id)
        )
    
    logger.info(f"✅ Created conditioning set '{name}' with {len(features)} features (ID: {feature_set_id})")
    return feature_set_id


async def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("Creating Template Conditioning Sets for Scenario Configuration")
    logger.info("=" * 80)
    
    db = CloudSQLManager()
    
    try:
        # Step 1: Get or create main project
        logger.info("\n📁 Step 1: Setting up main project...")
        project_id = await get_or_create_template_project(db)
        
        # Step 2: Create/update each conditioning set
        logger.info(f"\n🎯 Step 2: Creating/updating {len(CONDITIONING_SETS)} conditioning sets...")
        
        created_sets = []
        updated_sets = []
        for i, set_data in enumerate(CONDITIONING_SETS, 1):
            logger.info(f"\n[{i}/{len(CONDITIONING_SETS)}] Processing: {set_data['name']}")
            try:
                # Check if already exists
                async with db.connection() as conn:
                    existing = await conn.fetchrow("""
                        SELECT id, features, data_collectors FROM feature_sets
                        WHERE project_id = $1 
                        AND name = $2
                        AND set_type = 'conditioning'
                    """, db._to_uuid(project_id), set_data['name'])
                
                if existing:
                    # Update existing set with proper features/data_collectors
                    logger.info(f"🔄 Updating existing conditioning set '{set_data['name']}'")
                    
                    features_array = [
                        {
                            "id": feature["id"],
                            "source": feature["source"],
                            "name": feature["name"]
                        }
                        for feature in set_data["features"]
                    ]
                    
                    collectors_array = [
                        {
                            "source": collector["source"],
                            "api_key": collector.get("api_key")
                        }
                        for collector in set_data["data_collectors"]
                    ]
                    
                    async with db.connection() as conn:
                        await conn.execute("""
                            UPDATE feature_sets
                            SET features = $1::jsonb,
                                data_collectors = $2::jsonb,
                                description = $3
                            WHERE id = $4
                        """,
                            json.dumps(features_array),
                            json.dumps(collectors_array),
                            set_data["description"],
                            existing['id']
                        )
                    
                    updated_sets.append({
                        "name": set_data["name"],
                        "id": str(existing['id']),
                        "features_count": len(set_data["features"])
                    })
                    logger.info(f"✅ Updated '{set_data['name']}' with {len(set_data['features'])} features")
                else:
                    # Create new set
                    feature_set_id = await create_conditioning_set(db, project_id, set_data)
                    created_sets.append({
                        "name": set_data["name"],
                        "id": feature_set_id,
                        "features_count": len(set_data["features"])
                    })
            except Exception as e:
                logger.error(f"❌ Failed to process '{set_data['name']}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ CONDITIONING SETS OPERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nProject ID: {project_id}")
        logger.info(f"Created: {len(created_sets)} new conditioning sets")
        logger.info(f"Updated: {len(updated_sets)} existing conditioning sets")
        logger.info(f"Total: {len(created_sets) + len(updated_sets)}/{len(CONDITIONING_SETS)}\n")
        
        if created_sets:
            logger.info("Newly created conditioning sets:")
            for idx, set_info in enumerate(created_sets, 1):
                logger.info(f"  {idx}. {set_info['name']}")
                logger.info(f"     - ID: {set_info['id']}")
                logger.info(f"     - Features: {set_info['features_count']}")
        
        if updated_sets:
            logger.info("\nUpdated conditioning sets:")
            for idx, set_info in enumerate(updated_sets, 1):
                logger.info(f"  {idx}. {set_info['name']}")
                logger.info(f"     - ID: {set_info['id']}")
                logger.info(f"     - Features: {set_info['features_count']}")
        
        logger.info("\n📝 Next Steps:")
        logger.info("  1. Users can now create models using these conditioning sets with target sets")
        logger.info("  2. Train models to enable scenario simulation")
        logger.info("  3. Fetch data for conditioning sets using: POST /feature-sets/{id}/fetch-data")
        logger.info("  4. Configure scenarios using these conditioning features")
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

