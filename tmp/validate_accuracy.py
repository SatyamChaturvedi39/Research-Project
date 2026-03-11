import pandas as pd
import joblib
import numpy as np
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Paths
DATA_PATH = r'c:\SattyGithub\Research-Project\data\processed\player_features_v2_temporal.csv'
MODEL_PATH = r'c:\SattyGithub\Research-Project\models\player_multioutput_v2.pkl'
FEATURES_PATH = r'c:\SattyGithub\Research-Project\models\feature_names_v2.txt'
TARGETS_PATH = r'c:\SattyGithub\Research-Project\models\target_names_v2.txt'

def validate_model():
    print("\n" + "="*50)
    print("   NBA MODEL ACCURACY VALIDATION: 2024-25 SEASON")
    print("="*50)
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # 2. Filter for Test Set (Input is 2023-24, Ground Truth is target_next_* for 2024-25)
    test_df = df[(df['season'] == '2023-24') & (df['target_next_games'] > 5)].copy()
    
    print(f"Validated Samples: {len(test_df)} players (min 5 games in 2024-25)")
    
    # 3. Load Model and lists
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    with open(TARGETS_PATH, 'r') as f:
        targets = [line.strip() for line in f if line.strip()]

    # 4. Predict
    X_test = test_df[features]
    y_true = test_df[targets]
    y_pred = model.predict(X_test)
    
    if isinstance(y_pred, (list, tuple)):
        y_pred = np.column_stack(y_pred)
    
    y_pred_df = pd.DataFrame(y_pred, columns=targets, index=test_df.index)
    
    # 5. Metrics
    results = []
    for i, target in enumerate(targets):
        true_vals = y_true[target]
        pred_vals = y_pred_df[target]
        
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        r2 = r2_score(true_vals, pred_vals)
        
        results.append({
            "Metric": target.replace('target_next_', '').upper(),
            "MAE (Avg Error)": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2 Score": round(r2, 3)
        })
    
    # 6. Display Summary
    print("\nOVERALL ACCURACY SUMMARY:")
    print("-" * 50)
    print(pd.DataFrame(results).to_string(index=False))
    print("-" * 50)
    print("Interpretation: MAE shows the average absolute difference between prediction and reality.")
    print("An MAE of 1.40 for PPG means predictions were within ~1.4 points on average.")

    # 7. Random Samples
    print("\nSAMPLE PREDICTIONS vs ACTUALS (2024-25):")
    samples = test_df.sample(min(8, len(test_df)))
    for idx, row in samples.iterrows():
        name = row['player_name']
        print(f"\n> {name}:")
        p_ppg = y_pred_df.loc[idx, 'target_next_ppg']
        a_ppg = y_true.loc[idx, 'target_next_ppg']
        p_rpg = y_pred_df.loc[idx, 'target_next_rpg']
        a_rpg = y_true.loc[idx, 'target_next_rpg']
        print(f"  PPG: Pred {p_ppg:.1f} | Actual {a_ppg:.1f} (Δ {p_ppg-a_ppg:+.1f})")
        print(f"  RPG: Pred {p_rpg:.1f} | Actual {a_rpg:.1f} (Δ {p_rpg-a_rpg:+.1f})")

if __name__ == "__main__":
    validate_model()

if __name__ == "__main__":
    validate_model()
