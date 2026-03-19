import json
import re

notebook_path = r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Imports
        if 'import re' in source:
            source = source.replace('import re\n', '')
            source = source.replace('import re', '')
        
        # 2. Data loading
        if 'injury_df = pd.read_csv' in source:
            # Matches any read_csv for injury_df
            source = re.sub(r"injury_df = pd\.read_csv\(.*?\)\n?", "", source)
            source = re.sub(r"# Clean player names immediately\n?", "", source)
            source = re.sub(r"player_df\['player_name'\] = player_df\['player_name'\]\.str\.strip\(\)\n?", "", source)
            source = re.sub(r"injury_df\['Name'\] = injury_df\['Name'\]\.apply\(.*?\)\n?", "", source)
            source = re.sub(r"print\(f\"Injury data shape: {injury_df\.shape}\"\)\n?", "", source)
        
        # 3. Model loading
        if 'injury_model = joblib.load' in source:
            source = re.sub(r"injury_model = joblib\.load\(.*?\)\n?", "", source)

        # 4. Preprocessing
        if '# Prepare injury features' in source:
            # Matches the block and following merge
            source = re.sub(r"# Prepare injury features\n.*?\n.*?\n\n?", "", source, flags=re.DOTALL)
            source = re.sub(r"latest_player_data = latest_player_data\.merge\(latest_injury_stats.*?\)\.fillna\(0\)\n?", "", source)

        # 5. Projections
        if '# Map injury features for simulation' in source:
            replacement_proj_block = (
                "# Default injury features (to be integrated later)\n"
                "for col in ['minor_count', 'moderate_count', 'severe_count', 'prev_days_missed', 'prev_severe']:\n"
                "    proj_df[col] = 0\n"
            )
            source = re.sub(r"# Map injury features for simulation\n.*?\n.*?\n", replacement_proj_block, source, flags=re.DOTALL)

        # 6. Monte Carlo Engine
        if 'injury_model.predict_proba' in source:
            # Replace injury probability calculation with a constant
            source = re.sub(r"injury_probs = injury_model\.predict_proba\(.*?\)\n?", "injury_probs = np.full(len(roster), 0.05) # Placeholder injury risk\n", source)
            # Remove the feature list if it's there
            source = re.sub(r"injury_features = \[.*?\]\n?", "", source)

        # Split back into lines
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'].pop()
        else:
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully: removed injury dependencies.")
