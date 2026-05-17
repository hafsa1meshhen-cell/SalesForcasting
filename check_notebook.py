import json

# Read the notebook
with open('forecast_backup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Get the cell text
cell_text = ''.join(nb['cells'][0]['source']) if nb['cells'][0]['cell_type'] == 'code' else ''

# Check what we have
if 'parallel=None' in cell_text:
    print('✓ parallel=None found in notebook')
elif 'parallel="processes"' in cell_text:
    print('✗ parallel="processes" STILL in notebook - fixing now...')
    # Fix it by replacing in the lines
    for i, line in enumerate(nb['cells'][0]['source']):
        if 'parallel="processes"' in line:
            nb['cells'][0]['source'][i] = line.replace('parallel="processes"', 'parallel=None')
            print(f'Fixed line {i}')
    # Write back
    with open('forecast_backup.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('✓ Notebook saved with fix')
else:
    print('✓ No parallel="processes" found - may already be fixed')
