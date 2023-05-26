import csv

# if some lines have been deleted from the original file, use this script to fix the ids

file1 = "MaCoCu_after_rouge\MaCoCu-dataset-(all)2M+-filtered-06-with-ids.csv"
output_file = "MaCoCu_after_rouge\MaCoCu-dataset-(all)2M+-filtered-06-with-ids-corrected.csv"

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

        # remove the first column (old id-s)
        line_parts.pop(0)

        # Replace tab characters with escaped representation
        line_parts = [part.replace("\t", "\\t") for part in line_parts]

        writer.writerow([line_num - 1] + line_parts)

print("Done!")
