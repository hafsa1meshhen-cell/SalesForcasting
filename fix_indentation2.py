import re

with open('forecast.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the function definitions and helper sections
parts = []
current_pos = 0

# Keep everything up to safe_import_numpy_pandas() call
safe_call_match = re.search(r'\nsafe_import_numpy_pandas\(\)\n', content)
if safe_call_match:
    parts.append(content[:safe_call_match.end()])
    current_pos = safe_call_match.end()

# Add main function header
parts.append("\n\ndef main():\n")

# Find where if __name__ starts
name_main_match = re.search(r"\nif __name__ == '__main__':\n    main\(\)", content[current_pos:])
if name_main_match:
    # Get everything between safe_import and if __name__
    main_body = content[current_pos:current_pos + name_main_match.start()]
    
    # Indent every line by 4 spaces
    indented_lines = []
    for line in main_body.split('\n'):
        if line.strip():  # Non-empty line
            indented_lines.append('    ' + line)
        else:  # Empty line
            indented_lines.append(line)
    
    parts.append('\n'.join(indented_lines))
    parts.append("\n\n\nif __name__ == '__main__':\n    main()\n")
else:
    print("Could not find if __name__ guard")
    exit(1)

# Write the result
with open('forecast.py', 'w', encoding='utf-8') as f:
    f.write(''.join(parts))

print("File reconstructed successfully!")
