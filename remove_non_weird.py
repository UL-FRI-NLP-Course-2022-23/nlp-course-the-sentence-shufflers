import re

def is_slovene_alphabet(text):
    slovene_alphabet = set('abcčdefghijklmnoprsštuvzž.,?!-0123456789% \t')
    text = text.lower()
    return all(char in slovene_alphabet for char in text)

def remove_hyphen_and_weird_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.strip()
        line = line.replace(' - ', ' ')  # Remove ' - ' occurrences
        line = line.lstrip()  # Remove spaces at the beginning of the line
        if line and line[0] == '-':
            line = line[1:]  # Remove hyphen at the beginning of the line
        line = line.replace('"', '')  # Remove quotation marks
        if is_slovene_alphabet(line):
            print("cleaned line: ", line)
            cleaned_lines.append(line)

    with open(file_path, 'w') as file:
        file.write('\n'.join(cleaned_lines))

# Usage example
file_path = 'MaCoCu-dataset-(all)2M+-filtered-08-without-ids.csv'
remove_hyphen_and_weird_lines(file_path)
