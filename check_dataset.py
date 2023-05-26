import csv


# Print any line that is shorter than 3 fields

# Open the CSV file
with open('datasets/MaCoCu-dataset-(all)2M+-filtered-06-ids.csv', 'r', encoding='utf8') as file:
    reader = csv.reader(file, delimiter='\t', quotechar='"')
    for row in reader:
        if len(row) < 3 or len(row[1]) == 0 or len(row[2]) == 0:
            print(row)
