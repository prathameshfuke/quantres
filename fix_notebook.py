import json

with open('/Users/shura/Documents/quantres/quantizing_alpha_research.ipynb', 'r') as f:
    nb = json.load(f)

# Find and fix the cell with narrative issues
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Check if this is the experiment loop cell 
        if 'narrative_output, narrative_confidence = run_inference(narrative_context)' in source:
            print(f"Found issue in cell {i}")
            
            # Fix 1: Add narrative_prompt definition
            if 'narrative_prompt = narrative_context' not in source:
                insert_pos = source.find('# Run inference for both formats')
                if insert_pos > 0:
                    before = source[:insert_pos]
                    after = source[insert_pos:]
                    source = before + '    narrative_prompt = narrative_context  # Same as context for narrative format\n    \n    ' + after
                    print("✅ Added narrative_prompt definition")
            
            # Fix 2: Update the run_inference call to use narrative_prompt
            source = source.replace(
                'narrative_output, narrative_confidence = run_inference(narrative_context)',
                'narrative_output, narrative_confidence = run_inference(narrative_prompt)'
            )
            print("✅ Fixed narrative_output reference")
            
            # Update cell source back to list format
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

# Save the fixed notebook
with open('/Users/shura/Documents/quantres/quantizing_alpha_research.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("\n✅ Notebook fixed successfully!")
