import csv
import os.path

import evaluate

metric = evaluate.load("rouge")


# A script which calculates the rouge score and divides data according to the criteria

# rouge1: uni-gram (1-gram) based scoring
# rouge2: bi-gram (2-gram) based scoring
# rougeL: Longest common subsequence based scoring.
# rougeLSum: splits text using "\n"


def main(csv_file, csv_file_out):
    print("Open reader...")
    reader = csv.reader(csv_file, delimiter='\t')
    # Skip header
    header = next(reader, None)

    predictions = []
    references = []
    for line in reader:
        predictions.append(line[0])
        references.append(line[1])

    print("Calculating scores...")
    results = metric.compute(predictions=predictions, references=references, use_aggregator=False)
    # print(results)

    print("Filter out very similar pairs...")
    writer = csv.writer(csv_file_out, delimiter='\t')
    writer.writerow(header)
    for idx in range(len(results['rouge1'])):
        # print(results['rouge1'][idx], results['rouge2'][idx], results['rougeL'][idx], results['rougeLsum'][idx])
        if results['rouge1'][idx] < 0.8:
            writer.writerow([predictions[idx], references[idx]])

    csv_file.close()
    csv_file_out.close()


if __name__ == '__main__':
    file_in = 'datasets/MaCoCu-dataset-(all)2M+.csv'
    file = open(file_in, 'r', encoding='utf8')
    file_out = os.path.join(os.path.dirname(file_in), os.path.basename(file_in).split('.')[0] + '-filtered-08.' +
                            os.path.basename(file_in).split('.')[-1])
    out = open(file_out, 'w', encoding='utf8', newline='')
    main(file, out)
