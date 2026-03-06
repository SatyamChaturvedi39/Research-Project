import os

file_path = r'c:\MSAIML\research_proj\Research-Project\frontend\static\modal_styles.css'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replacements for modal_styles.css to use theme variables
replacements = {
    'background: linear-gradient(135deg, #1f2937 0%, #111827 100%)': 'background: var(--color-bg-dark)',
    'border-bottom: 1px solid rgba(255, 255, 255, 0.1)': 'border-bottom: 1px solid var(--color-border)',
    'color: white': 'color: var(--color-text)',
    'background: #374151': 'background: var(--color-bg-card)',
    'border: 2px solid #4b5563': 'border: 2px solid var(--color-border)',
    'background: #1f2937': 'background: var(--color-bg-dark)',
    'color: #9ca3af': 'color: var(--color-text-muted)',
    'background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)': 'background: var(--color-accent-primary)',
    'border-color: #60a5fa': 'border-color: var(--color-accent-secondary)',
    'border-color: #3b82f6': 'border-color: var(--color-accent-primary)',
    'box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3)': 'box-shadow: 0 8px 16px var(--color-accent-primary)',
}

# Add fallback for the color: white in headers etc
content = content.replace('color: white;', 'color: var(--color-text);')
content = content.replace('color: white', 'color: var(--color-text)')

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Modal styles sanitized for theme support.")
