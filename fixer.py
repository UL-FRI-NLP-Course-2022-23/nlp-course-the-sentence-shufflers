import csv

#script for removing [] and " " from id and text

input_file = "MaCoCu-dataset-(all)2M+-filtered-06-ids.csv"
output_file = "MaCoCu-dataset-(all)2M+-filtered-06-with-ids.csv"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", newline="", encoding="utf-8") as f_out:
    reader = csv.reader(f_in, delimiter="\t")
    writer = csv.writer(f_out, delimiter="\t")
    writer.writerow(["ID", "Original", "Paraphrase[Translation]"])

    for row in reader:
        # skip the header row
        if row[0] == "ID":
            continue

        # print(row)
        # Extract values from the row
        line_num = row[0].strip("[]") #strip the brackets
        text = row[1].strip('"') #strip the quotes
        
        #split the text into original and paraphrase
        original = text.split("\t")[0].strip()
        paraphrase = text.split("\t")[1].strip()

        # Write the processed row to the output CSV file
        writer.writerow([line_num, original, paraphrase])

        #print status message every 10000 lines
        if int(line_num) % 10000 == 0:
            print("Processing line {}...".format(line_num))

print("Done!")
