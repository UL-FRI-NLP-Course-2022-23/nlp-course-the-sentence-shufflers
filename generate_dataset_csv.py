import csv

# function to generate a dataset CSV file from two text files
# containing original and paraphrase sentences
# file1: path to the file containing original sentences
# file2: path to the file containing paraphrase sentences
# output_file: path to the output CSV file
# delimiter: delimiter to use in the output CSV file
# start_line: line number to start from in the input files
# end_line: line number to end at in the input files
# if end_line is None, the function will run until the end of the file

def generate_dataset_csv(file1, file2, output_file, delimiter_g="\t", start_line=0, end_line=None):
    # open input and output files
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2, open(output_file, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, delimiter=delimiter_g)
        writer.writerow(["Original", "Paraphrase[Translation]"])

        # if end_line is None, set it to infinity
        # so that the loop will run until the end of the file
        if end_line is None:
            end_line = float("inf")
        
        print("Generating dataset from lines {} to {}...".format(start_line, end_line))
        
        # iterate over lines in both input files simultaneously
        for line_num, (line1, line2) in enumerate(zip(f1, f2)):
            
            # skip over lines that come before the start of the range
            if line_num < start_line:
                continue
            
            # break out of the loop when you reach the end of the range
            if line_num >= end_line:
                break
            if line_num +1 % 10000 == 0:
                print("Processing line {}...".format(line_num))
            # write a row to the output CSV file
            writer.writerow([line1.strip(), line2.strip()])
    
    print("Done!")
    # return the output file name
    return output_file


# example usage

# input file names
file1 = "dataset_gen/MacoCu-half-orig-sl.txt"
file2 = "dataset_gen/MacoCu-half-tran-sl.txt"

# output file name
number_of_lines_in_k = 500
output_file = "dataset_gen/MaCoCu-dataset-{}k.csv".format(number_of_lines_in_k)

# call the function
generate_dataset_csv(file1, file2, output_file, end_line=number_of_lines_in_k*1000)
