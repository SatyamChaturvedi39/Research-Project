import json
import re

notebook_path = r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Join source lines into a single string for easier replacement
        source = "".join(cell['source'])
        
        # Patch imports
        if 'import pandas as pd' in source and 'import re' not in source:
            source = source.replace('import json', 'import json\nimport re')
        
        # Patch data loading
        if "injury_df = pd.read_csv('player_injury_score_2020_2025_cleaned.csv')" in source:
            target = "injury_df = pd.read_csv('player_injury_score_2020_2025_cleaned.csv')"
            replacement = (
                "injury_df = pd.read_csv('../data/processed/player_injury_score_2020_2025_cleaned.csv')\n\n"
                "# Clean player names immediately\n"
                "player_df['player_name'] = player_df['player_name'].str.strip()\n"
                "injury_df['Name'] = injury_df['Name'].apply(lambda x: re.sub(r'\\(.*?\\)', '', str(x)).strip())"
            )
            source = source.replace(target, replacement)
        
        # Split back into lines
        cell['source'] = [line + '\n' for line in source.split('\n')]
        # Remove the extra newline added by split if it's there
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'].pop()
        else:
            # Strip the trailing newline from the last actual line
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1) # Using indent=1 to match original style roughly

print("Notebook patched successfully using json.")
