def remove_duplicate_lines(file_path):
    line_dict = {}
    duplicate_lines = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    cleaned_lines = []
    for index, line in enumerate(lines):
        line = line.strip()
        if line in line_dict:
            duplicate_lines.append((line, line_dict[line], index))
        else:
            line_dict[line] = index
            cleaned_lines.append(line)

    with open(file_path, 'w') as file:
        file.write('\n'.join(cleaned_lines))

    return duplicate_lines


# Usage example
file_path = '' # put here the path to the file you want to remove duplicate lines from
duplicates = remove_duplicate_lines(file_path)

if duplicates:
    print("Duplicate lines found and removed:")
    for line, first_index, duplicate_index in duplicates:
        print(f"Line '{line}' was duplicated. Indexes: {first_index}, {duplicate_index}")
else:
    print("No duplicate lines found.")
