import nbformat as nbf
import re

notebook_path = r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Patch imports
        if 'import pandas as pd' in cell.source and 'import re' not in cell.source:
            cell.source = cell.source.replace('import json', 'import json\nimport re')
        
        # Patch data loading
        if 'injury_df = pd.read_csv(\'player_injury_score_2020_2025_cleaned.csv\')' in cell.source:
            target = "injury_df = pd.read_csv('player_injury_score_2020_2025_cleaned.csv')"
            replacement = (
                "injury_df = pd.read_csv('../data/processed/player_injury_score_2020_2025_cleaned.csv')\n\n"
                "# Clean player names immediately\n"
                "player_df['player_name'] = player_df['player_name'].str.strip()\n"
                "injury_df['Name'] = injury_df['Name'].apply(lambda x: re.sub(r'\\(.*?\\)', '', str(x)).strip())"
            )
            cell.source = cell.source.replace(target, replacement)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook patched successfully.")
