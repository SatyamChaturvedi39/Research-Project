import os

file_path = r'c:\MSAIML\research_proj\Research-Project\frontend\static\styles.css'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replacements to make CSS more theme-neutral
# Using more semantic variables instead of hardcoded RGBA values
replacements = {
    'color: rgba(255, 255, 255, 0.9)': 'color: var(--color-text)',
    'color: rgba(255, 255, 255, 0.75)': 'color: var(--color-text-muted)',
    'color: rgba(255, 255, 255, 0.35)': 'color: var(--color-text-muted)',
    'background: rgba(255, 255, 255, 0.04)': 'background: var(--color-bg-card)',
    'background: rgba(255, 255, 255, 0.05)': 'background: var(--color-bg-card)',
    'background: rgba(255, 255, 255, 0.06)': 'background: var(--color-bg-card)',
    'background: rgba(255, 255, 255, 0.1)': 'background: var(--color-bg-card)',
    'border: 1px solid rgba(255, 255, 255, 0.12)': 'border: 1px solid var(--color-border)',
    'border: 2px dashed rgba(255, 255, 255, 0.08)': 'border: 2px dashed var(--color-border)',
    'border-bottom: 1px solid rgba(255, 255, 255, 0.08)': 'border-bottom: 1px solid var(--color-border)',
    'background: rgba(13, 14, 33, 0.65)': 'background: var(--color-bg-dark)',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Styles sanitized for theme support.")
