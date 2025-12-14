input_path = 'NL2SQL_Cleaned.csv'
output_path = 'NL2SQL_Modified.csv'

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line_num, line in enumerate(infile, start=1):
        line = line.strip()

        # Find first comma
        first_comma = line.find(',')
        if first_comma != -1:
            # Insert '" ' after the first comma
            modified = line[:first_comma+1] + '" ' + line[first_comma+1:]

            # Append closing quote at the end if not already there
            if not modified.endswith('"'):
                modified += '"'
            
            outfile.write(modified + '\n')
        else:
            print(f"Line {line_num} skipped (no comma): {line}")
