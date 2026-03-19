import pandas as pd
import re

player_df = pd.read_csv(r'c:\MSAIML\research_proj\Research-Project\data\processed\player_features_v2_temporal.csv')
injury_df = pd.read_csv(r'c:\MSAIML\research_proj\Research-Project\data\processed\player_injury_score_2020_2025_cleaned.csv')

def clean_name(x):
    return re.sub(r'\(.*?\)', '', str(x)).strip()

p_names = set(player_df['player_name'].str.strip())
i_raw = set(injury_df['Name'])
i_clean = set(injury_df['Name'].apply(clean_name))

print("P:", len(p_names))
print("I_R:", len(i_raw))
print("I_C:", len(i_clean))
print("OR:", len(p_names.intersection(i_raw)))
print("OC:", len(p_names.intersection(i_clean)))
