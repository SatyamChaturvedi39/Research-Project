"""
Enhanced Data Preparation with Temporal Features
Replace cells in your 01_data_prep.ipynb with these code blocks
"""

# ========================================
# CELL 1: Imports and Setup
# ========================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Starting enhanced data preparation with temporal features...")


# ========================================
# CELL 2: Load and Merge All Seasons
# ========================================

# Load all 5 seasons
seasons = ['2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
season_labels = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

all_data = []

for season, label in zip(seasons, season_labels):
    df = pd.read_csv(f'../data/raw/nba_{season}_totals.csv')
    
    # Filter: min 10 games, min 100 minutes per season
    df = df[(df['G'] >= 10) & (df['MP'] >= 100)].copy()
    
    # Add season label
    df['season'] = label
    
    all_data.append(df)
    print(f"Loaded {label}: {len(df)} players")

# Combine all seasons
data = pd.concat(all_data, ignore_index=True)
print(f"\nTotal records: {len(data)}")


# ========================================
# CELL 3: Clean and Standardize Column Names
# ========================================

# First, check what columns we actually have
print("Columns in raw data:")
print(data.columns.tolist())

# Rename columns to consistent format
# Note: Check if 'Tm' or 'Team' exists in your data
rename_dict = {
    'Player': 'player_name',
    'Pos': 'position',
    'Age': 'age',
    'G': 'games_played',
    'GS': 'games_started',
    'MP': 'minutes_total',
    'FG': 'field_goals',
    'FGA': 'field_goal_attempts',
    'FG%': 'fg_pct',
    '3P': 'three_pointers',
    '3PA': 'three_point_attempts',
    '3P%': 'three_point_pct',
    '2P': 'two_pointers',
    '2PA': 'two_point_attempts',
    '2P%': 'two_point_pct',
    'FT': 'free_throws',
    'FTA': 'free_throw_attempts',
    'FT%': 'free_throw_pct',
    'ORB': 'offensive_rebounds',
    'DRB': 'defensive_rebounds',
    'TRB': 'total_rebounds',
    'AST': 'assists',
    'STL': 'steals',
    'BLK': 'blocks',
    'TOV': 'turnovers',
    'PF': 'personal_fouls',
    'PTS': 'points'
}

# Add team column (check both variants)
if 'Tm' in data.columns:
    rename_dict['Tm'] = 'team'
elif 'Team' in data.columns:
    rename_dict['Team'] = 'team'

data.rename(columns=rename_dict, inplace=True)

# Verify team column exists
if 'team' not in data.columns:
    raise ValueError("Team column not found! Please check your CSV file has 'Tm' or 'Team' column")

print("\nColumns after rename:")
print(data.columns.tolist())

# Handle traded players (2TM, 3TM) - keep only TOT row
data['is_total'] = data['team'].str.contains('TOT', na=False)
data['is_multi_team'] = data['team'].str.match(r'\d+TM', na=False)

# For players with TOT row, keep only TOT
# For players without TOT but multiple teams, keep first team
mask_tot = data['is_total']
mask_multi = data['is_multi_team'] & ~data['is_total']

data_clean = pd.concat([
    data[mask_tot],  # All TOT rows
    data[~mask_tot & ~mask_multi]  # Single team rows
], ignore_index=True)

# Drop helper columns
data_clean.drop(['is_total', 'is_multi_team'], axis=1, inplace=True)

print(f"After cleaning multi-team entries: {len(data_clean)} records")


# ========================================
# CELL 4: Calculate Per-Game Stats
# ========================================

# Calculate per-game averages
data_clean['minutes_per_game'] = data_clean['minutes_total'] / data_clean['games_played']
data_clean['points_per_game'] = data_clean['points'] / data_clean['games_played']
data_clean['rebounds_per_game'] = data_clean['total_rebounds'] / data_clean['games_played']
data_clean['assists_per_game'] = data_clean['assists'] / data_clean['games_played']
data_clean['steals_per_game'] = data_clean['steals'] / data_clean['games_played']
data_clean['blocks_per_game'] = data_clean['blocks'] / data_clean['games_played']
data_clean['turnovers_per_game'] = data_clean['turnovers'] / data_clean['games_played']

# Calculate efficiency metrics
data_clean['points_per_minute'] = data_clean['points'] / data_clean['minutes_total']
data_clean['field_goal_pct'] = data_clean['fg_pct'].fillna(0)
data_clean['free_throw_pct'] = data_clean['free_throw_pct'].fillna(0)

# True Shooting Percentage
data_clean['true_shooting_pct'] = np.where(
    (data_clean['field_goal_attempts'] + 0.44 * data_clean['free_throw_attempts']) > 0,
    data_clean['points'] / (2 * (data_clean['field_goal_attempts'] + 0.44 * data_clean['free_throw_attempts'])),
    0
)

# Usage rate approximation (simplified)
data_clean['usage_rate'] = (data_clean['field_goal_attempts'] + 
                             0.44 * data_clean['free_throw_attempts'] + 
                             data_clean['turnovers']) / data_clean['games_played']

# Assist and rebound rates
data_clean['assist_rate'] = np.where(
    data_clean['minutes_per_game'] > 0,
    data_clean['assists_per_game'] / data_clean['minutes_per_game'],
    0
)

data_clean['rebound_rate'] = np.where(
    data_clean['minutes_per_game'] > 0,
    data_clean['rebounds_per_game'] / data_clean['minutes_per_game'],
    0
)

print("Calculated per-game and efficiency stats")


# ========================================
# CELL 5: Create Temporal Features (LAG) - ALL 5 SEASONS
# ========================================

def create_temporal_features(df):
    """
    Add temporal features based on player history.
    For each player-season, add stats from previous 1-5 seasons.
    This uses ALL available history from the 5-season dataset.
    """
    
    # Sort by player and season
    df = df.sort_values(['player_name', 'season']).reset_index(drop=True)
    
    temporal_data = []
    
    print("\nProcessing temporal features by player...")
    total_players = df['player_name'].nunique()
    
    for idx, (name, group) in enumerate(df.groupby('player_name')):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total_players} players...")
        
        group = group.sort_values('season').reset_index(drop=True)
        
        for i in range(len(group)):
            row = group.iloc[i].to_dict()
            
            # Add lag features for ALL previous seasons (up to 5)
            for lag in [1, 2, 3, 4, 5]:
                if i >= lag:
                    prev = group.iloc[i - lag]
                    
                    # Key stats from previous seasons
                    row[f'ppg_lag{lag}'] = prev['points_per_game']
                    row[f'mpg_lag{lag}'] = prev['minutes_per_game']
                    row[f'rpg_lag{lag}'] = prev['rebounds_per_game']
                    row[f'apg_lag{lag}'] = prev['assists_per_game']
                    row[f'spg_lag{lag}'] = prev['steals_per_game']
                    row[f'bpg_lag{lag}'] = prev['blocks_per_game']
                    row[f'games_lag{lag}'] = prev['games_played']
                    row[f'ts_pct_lag{lag}'] = prev['true_shooting_pct']
                    row[f'fg_pct_lag{lag}'] = prev['field_goal_pct']
                else:
                    # Fill with NaN for players without sufficient history
                    row[f'ppg_lag{lag}'] = np.nan
                    row[f'mpg_lag{lag}'] = np.nan
                    row[f'rpg_lag{lag}'] = np.nan
                    row[f'apg_lag{lag}'] = np.nan
                    row[f'spg_lag{lag}'] = np.nan
                    row[f'bpg_lag{lag}'] = np.nan
                    row[f'games_lag{lag}'] = np.nan
                    row[f'ts_pct_lag{lag}'] = np.nan
                    row[f'fg_pct_lag{lag}'] = np.nan
            
            # Calculate trend features (change over 2, 3, and 4 years)
            if i >= 2:
                row['ppg_trend_2yr'] = (group.iloc[i]['points_per_game'] - 
                                        group.iloc[i-2]['points_per_game']) / 2
                row['mpg_trend_2yr'] = (group.iloc[i]['minutes_per_game'] - 
                                        group.iloc[i-2]['minutes_per_game']) / 2
            else:
                row['ppg_trend_2yr'] = 0
                row['mpg_trend_2yr'] = 0
            
            if i >= 3:
                row['ppg_trend_3yr'] = (group.iloc[i]['points_per_game'] - 
                                        group.iloc[i-3]['points_per_game']) / 3
            else:
                row['ppg_trend_3yr'] = 0
            
            if i >= 4:
                row['ppg_trend_4yr'] = (group.iloc[i]['points_per_game'] - 
                                        group.iloc[i-4]['points_per_game']) / 4
            else:
                row['ppg_trend_4yr'] = 0
            
            # Career statistics
            row['seasons_in_dataset'] = i + 1
            row['career_ppg_avg'] = group.iloc[:i+1]['points_per_game'].mean()
            row['career_ppg_std'] = group.iloc[:i+1]['points_per_game'].std() if i > 0 else 0
            row['career_games_avg'] = group.iloc[:i+1]['games_played'].mean()
            row['career_mpg_avg'] = group.iloc[:i+1]['minutes_per_game'].mean()
            
            # Years since peak PPG
            if i > 0:
                peak_idx = group.iloc[:i+1]['points_per_game'].idxmax()
                row['years_since_peak_ppg'] = i - (peak_idx - group.index[0])
                row['peak_ppg'] = group.iloc[:i+1]['points_per_game'].max()
            else:
                row['years_since_peak_ppg'] = 0
                row['peak_ppg'] = row['points_per_game']
            
            # Consistency metrics
            if i > 0:
                row['ppg_coefficient_variation'] = (group.iloc[:i+1]['points_per_game'].std() / 
                                                     group.iloc[:i+1]['points_per_game'].mean()) if group.iloc[:i+1]['points_per_game'].mean() > 0 else 0
            else:
                row['ppg_coefficient_variation'] = 0
            
            temporal_data.append(row)
    
    print(f"  Processed all {total_players} players")
    return pd.DataFrame(temporal_data)

# Apply temporal feature engineering
print("Creating temporal features with 5-season history...")
print(f"Starting with {len(data_clean)} player-season records")

df_temporal = create_temporal_features(data_clean)

# Track data at different filtering levels
print("\n" + "=" * 70)
print("DATA LOSS TRACKING")
print("=" * 70)

print(f"\nTotal player-seasons after temporal feature creation: {len(df_temporal)}")
print(f"Unique players: {df_temporal['player_name'].nunique()}")

# Count players by seasons available
for min_seasons in [1, 2, 3, 4, 5]:
    count = len(df_temporal[df_temporal['seasons_in_dataset'] >= min_seasons])
    print(f"  - Players with {min_seasons}+ seasons: {count} ({count/len(df_temporal)*100:.1f}%)")

# Strategy: Keep players with at least 2 previous seasons for good predictions
# But preserve all data - use imputation for missing lag features
print("\nFiltering strategy: Require at least 2 previous seasons (lag2 exists)")
df_temporal_clean = df_temporal.dropna(subset=['ppg_lag2', 'mpg_lag2'])

data_loss_pct = (1 - len(df_temporal_clean) / len(df_temporal)) * 100

print(f"\nAfter requiring 2+ season history:")
print(f"  Records: {len(df_temporal_clean)} (lost {data_loss_pct:.1f}%)")
print(f"  Unique players: {df_temporal_clean['player_name'].nunique()}")
print(f"  Seasons covered: {df_temporal_clean['season'].min()} to {df_temporal_clean['season'].max()}")

# Show season distribution
print("\nDistribution by season:")
for season in sorted(df_temporal_clean['season'].unique()):
    count = len(df_temporal_clean[df_temporal_clean['season'] == season])
    print(f"  {season}: {count} player-seasons")



# ========================================
# CELL 6: Create Target Variables
# ========================================

def create_targets(df):
    """
    Create next-season targets for prediction.
    For each player-season, add next season's stats as targets.
    """
    
    df = df.sort_values(['player_name', 'season']).reset_index(drop=True)
    
    target_data = []
    
    for name, group in df.groupby('player_name'):
        group = group.sort_values('season').reset_index(drop=True)
        
        # Only keep seasons that have a next season
        for i in range(len(group) - 1):
            row = group.iloc[i].to_dict()
            next_season = group.iloc[i + 1]
            
            # Target: next season's performance
            row['target_next_ppg'] = next_season['points_per_game']
            row['target_next_rpg'] = next_season['rebounds_per_game']
            row['target_next_apg'] = next_season['assists_per_game']
            row['target_next_mpg'] = next_season['minutes_per_game']
            row['target_next_ts_pct'] = next_season['true_shooting_pct']
            row['target_next_games'] = next_season['games_played']
            
            target_data.append(row)
    
    return pd.DataFrame(target_data)

# Create targets
print("Creating target variables...")
df_final = create_targets(df_temporal_clean)

print(f"Final dataset with targets: {len(df_final)} training samples")
print(f"Time span: {df_final['season'].min()} to {df_final['season'].max()}")


# ========================================
# CELL 7: Select Final Features and Save
# ========================================

# Define feature columns for modeling
base_features = [
    'age', 'games_played', 'minutes_per_game',
    'points_per_game', 'rebounds_per_game', 'assists_per_game',
    'steals_per_game', 'blocks_per_game', 'turnovers_per_game',
    'field_goal_pct', 'free_throw_pct', 'true_shooting_pct',
    'points_per_minute', 'usage_rate', 'assist_rate', 'rebound_rate'
]

lag_features = [
    # Points per game - all 5 lags
    'ppg_lag1', 'ppg_lag2', 'ppg_lag3', 'ppg_lag4', 'ppg_lag5',
    # Minutes per game - all 5 lags
    'mpg_lag1', 'mpg_lag2', 'mpg_lag3', 'mpg_lag4', 'mpg_lag5',
    # Rebounds per game
    'rpg_lag1', 'rpg_lag2', 'rpg_lag3',
    # Assists per game
    'apg_lag1', 'apg_lag2', 'apg_lag3',
    # Other key stats
    'spg_lag1', 'bpg_lag1',
    'games_lag1', 'games_lag2', 'games_lag3',
    'ts_pct_lag1', 'ts_pct_lag2', 'ts_pct_lag3',
    'fg_pct_lag1', 'fg_pct_lag2'
]

trend_features = [
    'ppg_trend_2yr', 'ppg_trend_3yr', 'ppg_trend_4yr',
    'mpg_trend_2yr',
    'seasons_in_dataset', 'years_since_peak_ppg', 'peak_ppg',
    'career_ppg_avg', 'career_ppg_std',
    'career_games_avg', 'career_mpg_avg',
    'ppg_coefficient_variation'
]

target_features = [
    'target_next_ppg', 'target_next_rpg', 'target_next_apg',
    'target_next_mpg', 'target_next_ts_pct', 'target_next_games'
]

metadata_features = ['player_name', 'team', 'season', 'position']

# Select columns
all_features = metadata_features + base_features + lag_features + trend_features + target_features

# Ensure all columns exist
available_features = [f for f in all_features if f in df_final.columns]
df_export = df_final[available_features].copy()

# Fill any remaining NaN values
df_export = df_export.fillna(0)

# Save to CSV
output_path = '../data/processed/player_features_v2_temporal.csv'
df_export.to_csv(output_path, index=False)

print(f"\nSaved enhanced dataset to: {output_path}")
print(f"Shape: {df_export.shape}")
print(f"Features: {len(base_features + lag_features + trend_features)}")
print(f"Targets: {len(target_features)}")

# Display summary
print("\nDataset Summary:")
print(df_export.describe())

print("\nSample row:")
print(df_export[metadata_features + ['ppg_lag1', 'ppg_lag2', 'target_next_ppg']].head())
