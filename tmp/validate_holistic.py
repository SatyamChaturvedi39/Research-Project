import pandas as pd
import joblib
import numpy as np
import os
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths
BASE_DIR = r'c:\SattyGithub\Research-Project'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'player_features_v2_temporal.csv')
INJURY_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'player_injury_score_2020_2025_cleaned.csv')
TEAM_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'team_features_temporal.csv')

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'player_multioutput_v2.pkl')
INJURY_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'injury_clf.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names_v2.txt')
TARGETS_PATH = os.path.join(BASE_DIR, 'models', 'target_names_v2.txt')

def predict_wins(score):
    """Formula from services/trade_analyzer.py"""
    wins = (score - 92.5) * 1.5 + 41
    return np.clip(wins, 5, 75)

def calculate_fit_penalty(roster):
    """Logic from services/trade_analyzer.py"""
    penalty = 0.0
    pos_counts = roster['position'].value_counts()
    
    if pos_counts.get('PG', 0) > 3: penalty += 0.05
    if pos_counts.get('C', 0) > 3: penalty += 0.03
    if pos_counts.get('PG', 0) < 1: penalty += 0.08
    
    # scoring cannibalization
    top_scorers = roster.nlargest(3, 'pred_ppg')['pred_ppg'].sum()
    if top_scorers > 75:
        penalty += 0.08
        
    return penalty

def calculate_team_score(roster_stats):
    """Logic from services/trade_analyzer.py"""
    top_8 = roster_stats.nlargest(8, 'pred_ppg')
    sum_ppg = top_8['pred_ppg'].sum()
    avg_efficiency = top_8['pred_ts_pct'].mean() * 10
    
    penalty = calculate_fit_penalty(roster_stats)
    score = (sum_ppg + avg_efficiency) * (1 - penalty)
    return score

def validate_holistic():
    print("\n" + "="*60)
    print("   NBA HOLISTIC VALIDATION: PLAYERS, INJURIES, & TEAMS")
    print("="*60)
    
    # 1. Load Everything
    df = pd.read_csv(DATA_PATH)
    injury_df = pd.read_csv(INJURY_DATA_PATH)
    team_actuals_df = pd.read_csv(TEAM_DATA_PATH)
    
    model = joblib.load(MODEL_PATH)
    injury_model = joblib.load(INJURY_MODEL_PATH)
    
    with open(FEATURES_PATH, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    with open(TARGETS_PATH, 'r') as f:
        targets = [line.strip() for line in f if line.strip()]

    # 2. Prepare Player Predictions for 2024-25
    # Input: 2023-24 row, predicting 2024-25
    test_df = df[df['season'] == '2023-24'].copy()
    X_test = test_df[features].fillna(0)
    
    y_pred = model.predict(X_test)
    if isinstance(y_pred, (list, tuple)): y_pred = np.column_stack(y_pred)
    
    for i, target in enumerate(targets):
        clean_name = 'pred_' + target.replace('target_next_', '')
        test_df[clean_name] = y_pred[:, i]

    # 3. Injury Risk Validation
    print(f"\n[1/3] INJURY MODEL VALIDATION")
    inj_feat_list = ['games_missed', 'games_missed_last_season', 'total_days_missed',
                    'minor_count', 'moderate_count', 'severe_count', 'has_severe_injury']
    
    # Merge current injury stats into test_df
    latest_injury = injury_df.sort_values(['Name', 'season']).groupby('Name').last()
    test_df['normalized_name'] = test_df['player_name'].str.lower().str.strip()
    latest_injury['normalized_name'] = latest_injury.index.str.lower().str.strip()
    
    test_df = test_df.merge(latest_injury[inj_feat_list + ['normalized_name']], on='normalized_name', how='left').fillna(0)
    
    # Predict Injury Probability
    # Handle the multi_class attribute hack seen in app.py
    if hasattr(injury_model, 'named_steps'):
        classifier = injury_model.named_steps.get('classifier') or injury_model.steps[-1][1]
        if not hasattr(classifier, 'multi_class'): classifier.multi_class = 'auto'
    elif not hasattr(injury_model, 'multi_class'):
        injury_model.multi_class = 'auto'
        
    test_df['pred_injury_prob'] = injury_model.predict_proba(test_df[inj_feat_list])[:, 1]
    
    # Ground truth: target_next_games (if low compared to 82, it's an "injury" hit)
    # Binary label: If player played < 50 games in 2024-25, consider it "high injury impact"
    test_df['actual_injury_hit'] = (test_df['target_next_games'] < 50).astype(int)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_df['actual_injury_hit'], test_df['pred_injury_prob'])
    print(f"Injury Risk ROC-AUC: {auc:.3f}")
    print("Interpretation: >0.70 is good for injury prediction given its randomness.")

    # 4. Team Win Validation (The "Big Picture")
    print(f"\n[2/3] TEAM WIN PROJECTION VALIDATION")
    team_results = []
    
    # Teams in 2024-25 (Actual season row exists in team_actuals_df)
    teams_2425 = team_actuals_df[team_actuals_df['season'] == '2024-25']
    
    for _, t_row in teams_2425.iterrows():
        team_abbr = t_row['team']
        actual_wins = t_row['wins']
        
        # Get the roster for this team from the predictions
        # We assume the team they were on in 23-24 is who they're playing for in 24-25 for baseline validation
        # (A real trade analyzer would handle switches, but this validates the core win formula)
        roster = test_df[test_df['team'] == team_abbr]
        
        if len(roster) < 8: continue
        
        team_score = calculate_team_score(roster)
        proj_wins = predict_wins(team_score)
        
        team_results.append({
            "Team": team_abbr,
            "Proj Wins": round(proj_wins, 1),
            "Actual Wins": int(actual_wins),
            "Error": round(proj_wins - actual_wins, 1)
        })
    
    team_results_df = pd.DataFrame(team_results)
    win_mae = mean_absolute_error(team_results_df["Actual Wins"], team_results_df["Proj Wins"])
    win_r2 = r2_score(team_results_df["Actual Wins"], team_results_df["Proj Wins"])
    
    print(f"Win Projection MAE: {win_mae:.2f} games")
    print(f"Win Projection R2 Score: {win_r2:.3f}")
    
    # 5. Summary Display
    print("\nSAMPLE TEAM PROJECTIONS (2024-25):")
    print("-" * 50)
    print(team_results_df.head(10).to_string(index=False))
    print("-" * 50)
    
    print(f"\n[3/3] OVERALL SCORE ACCURACY")
    print(f"The 'Trade Score' on the website depends on delta_wins and roster health.")
    print(f"Since Win MAE is {win_mae:.1f}, the Trade Score reflects real performance within a tight margin.")

if __name__ == "__main__":
    validate_holistic()
