import json

notebook_path = r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Cleanup glitch in run_simulation
        if "):[:, 1]" in source or ")\n):[:, 1]" in source or "[:, 1]" in source:
             # More specific replacement to avoid breaking other things
             source = source.replace(")\n):[:, 1]\n", "\n")
             source = source.replace(")[:, 1]\n", "")
        
        # Split back into lines
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'].pop()
        else:
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook final cleanup complete.")
