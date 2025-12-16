with open('forecast.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where main() starts
main_start = None
for i, line in enumerate(lines):
    if line.strip() == 'def main():':
        main_start = i
        break

if main_start is None:
    print("Could not find def main():")
    exit(1)

# Process the file
new_lines = []
new_lines.extend(lines[:main_start+1])  # Keep everything up to and including def main():

# Find where if __name__ starts
name_main_start = None
for i in range(len(lines)-1, main_start, -1):
    if "if __name__ == '__main__':" in lines[i]:
        name_main_start = i
        break

if name_main_start is None:
    print("Could not find if __name__ == '__main__':")
    exit(1)

# Indent everything between def main(): and if __name__
for i in range(main_start+1, name_main_start):
    line = lines[i]
    # Don't indent empty lines or lines that are already comments at start
    if line.strip() == '':
        new_lines.append(line)
    else:
        # Add 4 spaces of indentation
        new_lines.append('    ' + line)

# Add the if __name__ part
new_lines.extend(lines[name_main_start:])

# Write back
with open('forecast.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed indentation successfully!")
