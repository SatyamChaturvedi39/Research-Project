import pandas as pd
import pickle
import joblib
import json
import os

print("=== CHECKING DATASETS ===")
datasets = [
    'data/processed/player_features_v2_temporal.csv',
    'data/processed/player_injury_score_2020_2025_cleaned.csv',
    'data/processed/team_features_temporal.csv'
]

for ds in datasets:
    if os.path.exists(ds):
        try:
            df = pd.read_csv(ds)
            print(f"\n--- {ds} ---")
            print(f"Shape: {df.shape}")
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]
            if not missing_cols.empty:
                print(f"Missing Values (Top 5 columns):\n{missing_cols.sort_values(ascending=False).head(5)}")
            else:
                print("No missing values.")
            
            # Print a few continuous stat columns to check scaling/validity
            if 'points_per_game' in df.columns:
                print(f"PPG Summary: min={df['points_per_game'].min():.1f}, max={df['points_per_game'].max():.1f}, mean={df['points_per_game'].mean():.1f}")
            if 'injury_score' in df.columns:
                print(f"Injury Score Summary: min={df['injury_score'].min():.1f}, max={df['injury_score'].max():.1f}, mean={df['injury_score'].mean():.1f}")
            if 'wins' in df.columns:
                print(f"Wins Summary: min={df['wins'].min():.1f}, max={df['wins'].max():.1f}, mean={df['wins'].mean():.1f}")
                
            # If 'season' or dates exist
            if 'season' in df.columns:
                print(f"Seasons: {df['season'].unique()}")
                
        except Exception as e:
            print(f"Error loading {ds}: {e}")
    else:
        print(f"\n[NOT FOUND] {ds}")

print("\n=== CHECKING MODELS & ARTIFACTS ===")
models = [
    'models/player_multioutput_v2.pkl',
    'models/injury_clf.pkl',
    'models/shap_explainers_v2.pkl',
    'models/model_metadata_v2.json'
]

for m in models:
    if os.path.exists(m):
        print(f"\n--- {m} ---")
        try:
            if m.endswith('.pkl'):
                try:
                    obj = joblib.load(m)
                except:
                    with open(m, 'rb') as f:
                        obj = pickle.load(f)
                
                print(f"Type: {type(obj)}")
                
            elif m.endswith('.json'):
                with open(m, 'r') as f:
                    data = json.load(f)
                print(f"Keys: {list(data.keys())}")
                if 'performance' in data:
                    print(f"Performance Metrics available for: {list(data['performance'].keys())}")
        except Exception as e:
            print(f"Error loading {m}: {e}")
    else:
        print(f"\n[NOT FOUND] {m}")

print("\n=== DONE ===")
