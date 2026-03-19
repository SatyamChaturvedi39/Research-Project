import json

notebook_path = r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Cleanup glitch in load-data
        if "', '', str(x)).strip())" in source:
            source = source.replace("', '', str(x)).strip())\n", "")
        
        # Ensure injury_probs is defined in run_simulation
        if 'def run_simulation(roster, n_sims=1000):' in source:
            if 'injury_probs =' not in source:
                # Insert it at the top of the function
                source = source.replace(
                    "def run_simulation(roster, n_sims=1000):",
                    "def run_simulation(roster, n_sims=1000):\n    injury_probs = np.full(len(roster), 0.05) # Placeholder risk"
                )
        
        # Split back into lines
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'].pop()
        else:
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook cleanup complete.")
