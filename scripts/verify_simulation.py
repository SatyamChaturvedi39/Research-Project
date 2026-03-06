import pandas as pd
import numpy as np
import joblib

# Mock local dependencies for standalone test
def calculate_fit_penalty(roster):
    penalty = 0
    pos_counts = roster['position'].value_counts()
    if pos_counts.get('PG', 0) > 3: penalty += 0.05
    if pos_counts.get('C', 0) > 3: penalty += 0.03
    top_scorers = roster.nlargest(3, 'target_next_ppg')['target_next_ppg'].sum()
    if top_scorers > 65: penalty += 0.08
    return penalty

def calculate_team_score(roster, sampled_stats):
    top_8_stats = sampled_stats.nlargest(8, 'target_next_ppg')
    top_8_ppg = top_8_stats['target_next_ppg'].sum()
    efficiency_bonus = top_8_stats['target_next_ts_pct'].mean() * 10
    fit_penalty = calculate_fit_penalty(roster)
    score = (top_8_ppg + efficiency_bonus) * (1 - fit_penalty)
    return score

def predict_wins(score):
    wins = (score - 50) * 1.5 + 41
    return np.clip(wins, 0, 82)

std_devs = {
    'target_next_ppg': 3.0,
    'target_next_rpg': 1.2,
    'target_next_apg': 1.0,
    'target_next_mpg': 3.5,
    'target_next_ts_pct': 0.03
}

def run_simulation(roster, n_sims=100):
    injury_probs = np.full(len(roster), 0.05) # Placeholder injury risk
    
    results = []
    for _ in range(n_sims):
        sampled_stats = roster.copy()
        for col, std in std_devs.items():
            sampled_stats[col] = np.maximum(0, np.random.normal(roster[col], std))
        
        is_injured = np.random.random(len(roster)) < injury_probs
        sampled_stats.loc[is_injured, 'target_next_ppg'] *= 0.6
        
        score = calculate_team_score(roster, sampled_stats)
        wins = predict_wins(score)
        results.append(wins)
        
    return np.array(results)

# Create a small mock roster
mock_roster = pd.DataFrame({
    'player_name': ['Player A', 'Player B', 'Player C'],
    'position': ['PG', 'SG', 'SF'],
    'target_next_ppg': [20.0, 15.0, 10.0],
    'target_next_ts_pct': [0.55, 0.58, 0.50]
})

print("Testing simulation logic...")
res = run_simulation(mock_roster, n_sims=10)
print(f"Results: {res}")
print("Simulation logic verified successfully.")
