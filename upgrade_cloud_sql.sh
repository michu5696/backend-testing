#!/bin/bash
# Script to upgrade Cloud SQL instance tier
# Usage: ./upgrade_cloud_sql.sh [new_tier]
# Example: ./upgrade_cloud_sql.sh db-n1-standard-1

set -e

INSTANCE_NAME="sablier-db"
CURRENT_TIER=$(gcloud sql instances describe $INSTANCE_NAME --format="value(settings.tier)" 2>/dev/null)

echo "🔍 Current Cloud SQL Instance Configuration:"
echo "   Instance: $INSTANCE_NAME"
echo "   Current Tier: $CURRENT_TIER"
echo ""

# Default to db-n1-standard-1 if not specified
NEW_TIER=${1:-"db-n1-standard-1"}

echo "📊 Recommended Tiers for Production:"
echo "   • db-n1-standard-1: ~250 connections, 3.75GB RAM, 1 vCPU (~$50/month)"
echo "   • db-n1-standard-2: ~250 connections, 7.5GB RAM, 2 vCPU (~$100/month)"
echo "   • db-n1-standard-4: ~250 connections, 15GB RAM, 4 vCPU (~$200/month)"
echo ""

if [ "$CURRENT_TIER" == "$NEW_TIER" ]; then
    echo "⚠️  Instance is already at tier: $NEW_TIER"
    exit 0
fi

echo "⚠️  WARNING: This will upgrade the instance tier from $CURRENT_TIER to $NEW_TIER"
echo "   • The upgrade will cause a brief downtime (typically 1-5 minutes)"
echo "   • All connections will be dropped during the upgrade"
echo "   • Make sure no critical operations are running"
echo ""
read -p "Do you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Upgrade cancelled"
    exit 1
fi

echo ""
echo "🚀 Upgrading Cloud SQL instance tier..."
echo "   From: $CURRENT_TIER"
echo "   To: $NEW_TIER"
echo ""

# Perform the upgrade
gcloud sql instances patch $INSTANCE_NAME \
    --tier=$NEW_TIER \
    --quiet

echo ""
echo "✅ Upgrade initiated!"
echo ""
echo "📝 Next Steps:"
echo "   1. Monitor the upgrade status:"
echo "      gcloud sql operations list --instance=$INSTANCE_NAME"
echo ""
echo "   2. Check instance status:"
echo "      gcloud sql instances describe $INSTANCE_NAME --format='value(state)'"
echo ""
echo "   3. After upgrade, verify connection limits:"
echo "      gcloud sql instances describe $INSTANCE_NAME --format='value(settings.tier)'"
echo ""
echo "   4. Update connection pool in code if needed (currently set to 10 per instance)"
echo ""

