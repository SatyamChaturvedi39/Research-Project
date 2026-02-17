"""
Populate MongoDB with NBA player data from CSV
Converts player_features_v2_temporal.csv to MongoDB documents
"""
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

# Get MongoDB URI from environment
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    print("ERROR: MONGODB_URI not set in .env file")
    print("Please edit .env and add your MongoDB connection string")
    sys.exit(1)

DATABASE_NAME = os.getenv('DATABASE_NAME', 'nba_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'players')


# Path to CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "player_features_v2_temporal.csv")

# ============================================================================
# Load and Transform Data
# ============================================================================

print("=" * 80)
print("NBA PLAYER DATA - MongoDB Population Script")
print("=" * 80)

# Load CSV
print(f"\n[1/5] Loading data from {CSV_PATH}...")
if not os.path.exists(CSV_PATH):
    print(f"✗ Error: CSV file not found at {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"✓ Loaded {len(df)} player-season records")

# Group by player to create historical documents
print("\n[2/5] Grouping data by player...")
player_groups = df.groupby('player_name')
print(f"✓ Found {len(player_groups)} unique players")

# ============================================================================
# Build MongoDB Documents
# ============================================================================

print("\n[3/5] Building MongoDB documents...")

documents = []

for player_name, player_df in player_groups:
    # Sort by season
    player_df = player_df.sort_values('season')
    
    # Get latest info
    latest = player_df.iloc[-1]
    
    # Build stats array (one entry per season)
    stats_array = []
    for _, row in player_df.iterrows():
        season_stats = {
            "season": row['season'],
            "team": row['team'],
            "age": int(row['age']) if pd.notna(row['age']) else 0,
            "games_played": int(row['games_played']) if pd.notna(row['games_played']) else 0,
            
            # Per-game stats
            "minutes_per_game": float(row['minutes_per_game']) if pd.notna(row['minutes_per_game']) else 0.0,
            "points_per_game": float(row['points_per_game']) if pd.notna(row['points_per_game']) else 0.0,
            "rebounds_per_game": float(row['rebounds_per_game']) if pd.notna(row['rebounds_per_game']) else 0.0,
            "assists_per_game": float(row['assists_per_game']) if pd.notna(row['assists_per_game']) else 0.0,
            "steals_per_game": float(row['steals_per_game']) if pd.notna(row['steals_per_game']) else 0.0,
            "blocks_per_game": float(row['blocks_per_game']) if pd.notna(row['blocks_per_game']) else 0.0,
            "turnovers_per_game": float(row['turnovers_per_game']) if pd.notna(row['turnovers_per_game']) else 0.0,
            
            # Shooting percentages
            "field_goal_pct": float(row['field_goal_pct']) if pd.notna(row['field_goal_pct']) else 0.0,
            "free_throw_pct": float(row['free_throw_pct']) if pd.notna(row['free_throw_pct']) else 0.0,
            "true_shooting_pct": float(row['true_shooting_pct']) if pd.notna(row['true_shooting_pct']) else 0.0,
            
            # Advanced stats
            "points_per_minute": float(row['points_per_minute']) if pd.notna(row['points_per_minute']) else 0.0,
            "usage_rate": float(row['usage_rate']) if pd.notna(row['usage_rate']) else 0.0,
            "assist_rate": float(row['assist_rate']) if pd.notna(row['assist_rate']) else 0.0,
            "rebound_rate": float(row['rebound_rate']) if pd.notna(row['rebound_rate']) else 0.0,
        }
        stats_array.append(season_stats)
    
    # Build document
    doc = {
        "player_name": player_name,
        "team": latest['team'],
        "position": latest.get('position', 'N/A') if 'position' in latest.index else 'N/A',
        "age": int(latest['age']) if pd.notna(latest['age']) else 0,
        "seasons_count": len(stats_array),
        "stats": stats_array,
        "search_name": player_name.lower(),  # For case-insensitive search
        "last_updated": datetime.utcnow()
    }
    
    documents.append(doc)

print(f"✓ Created {len(documents)} player documents")

# ============================================================================
# Connect to MongoDB
# ============================================================================

print("\n[4/5] Connecting to MongoDB...")
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Test connection
    client.server_info()
    print(f"✓ Connected to MongoDB successfully")
    
except Exception as e:
    print(f"✗ Connection failed: {str(e)}")
    print("\nTroubleshooting:")
    print("  - Check if MongoDB is running (local): net start MongoDB")
    print("  - Verify connection string (Atlas)")
    print("  - Check network access settings (Atlas - allow 0.0.0.0/0)")
    sys.exit(1)

# ============================================================================
# Insert Data
# ============================================================================

print("\n[5/5] Inserting data into MongoDB...")

# Drop existing collection (fresh start)
collection.drop()
print("✓ Cleared existing data")

# Insert all documents
result = collection.insert_many(documents)
print(f"✓ Inserted {len(result.inserted_ids)} player documents")

# Create indexes for performance
print("\nCreating indexes...")
collection.create_index("player_name")
collection.create_index("search_name")
collection.create_index("team")
print("✓ Indexes created")

# ============================================================================
# Verification
# ============================================================================

print("\n" + "=" * 80)
print("DATA VERIFICATION")
print("=" * 80)

# Count documents
total_docs = collection.count_documents({})
print(f"\nTotal players in database: {total_docs}")

# Sample document
sample = collection.find_one()
if sample:
    print(f"\nSample player: {sample['player_name']}")
    print(f"  Team: {sample['team']}")
    print(f"  Age: {sample['age']}")
    print(f"  Seasons: {sample['seasons_count']}")
    print(f"  Latest season: {sample['stats'][-1]['season']}")
    print(f"  Latest PPG: {sample['stats'][-1]['points_per_game']:.1f}")

# Test search
test_search = collection.find_one({"search_name": {"$regex": "james"}})
if test_search:
    print(f"\nSearch test (James): ✓ Found {test_search['player_name']}")

print("\n" + "=" * 80)
print("✅ DATABASE SETUP COMPLETE!")
print("=" * 80)
print(f"\nConnection string: {MONGODB_URI}")
print(f"Database: {DATABASE_NAME}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Total players: {total_docs}")
print("\nNext step: Update frontend/app.py MONGODB_URI variable")
print("=" * 80)

client.close()
