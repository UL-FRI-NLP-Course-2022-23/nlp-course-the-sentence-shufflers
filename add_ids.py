import csv


# Script to add input_ids to pairs of sentences

file1 = "./MaCoCu-datasets/MaCoCu-dataset-(all)2M+-filtered-06.csv"
output_file = "datasets/MaCoCu-dataset-(all)2M+-filtered-06-ids.csv"

with open(file1, "r", encoding="utf-8") as f1, open(output_file, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
    writer.writerow(["ID", "Original", "Paraphrase[Translation]"])

    # add id as first column
    for line_num, line in enumerate(f1):
        if line_num == 0:
            continue
        # print a status message every 10000 lines
        if line_num % 10000 == 0:
            print("Processing line {}...".format(line_num))

        line_parts = line.strip().split("\t")

        # assuming a file with no ids has 2 columns
        if len(line_parts) < 2:
            print("Skipping line {}...".format(line_num))
            continue  # skip lines with no paraphrase

        # Replace tab characters with escaped representation
        line_parts = [part.replace("\t", "\\t") for part in line_parts]

        writer.writerow([line_num - 1] + line_parts)

print("Done!")
