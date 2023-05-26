import csv
import os.path

import evaluate
import numpy as np
from parascore import ParaScorer

scorer = ParaScorer(lang="sl", model_type='bert-base-uncased')

metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")
metric_bertscore = evaluate.load("bertscore")
metric_wer = evaluate.load("wer")
metric_meteor = evaluate.load("meteor")
metric_gleu = evaluate.load("google_bleu")


def calculate_matrices(csv_file_in, csv_file_out):
    # calculate metrics and write the results to csv file
    print("Open reader...")
    reader = csv.reader(csv_file_in, delimiter='\t')

    predictions = []
    references = []
    for line in reader:
        predictions.append(line[1])
        references.append(line[0])

    print("Calculating scores...")
    results_bleu = metric_bleu.compute(predictions=predictions, references=references)
    results_rouge = metric_rouge.compute(predictions=predictions, references=references, use_aggregator=True)
    results_bertscore = metric_bertscore.compute(predictions=predictions, references=references, lang="sl")
    results_wer = metric_wer.compute(predictions=predictions, references=references)
    results_meteor = metric_meteor.compute(predictions=predictions, references=references)
    results_gleu = metric_gleu.compute(predictions=predictions, references=references)

    parascore_scores = []
    for source, candidate in zip(references, predictions):
        score = scorer.free_score([candidate], [source])
        parascore_scores.append(score[0].item())

    results_parascore = np.mean(parascore_scores)

    # Write the results to console
    # print("Writing results to console...")
    # print("bleu:", results_bleu['bleu'])
    # print("rouge1:", results_rouge['rouge1'])
    # print("rouge2:", results_rouge['rouge2'])
    # print("rougeL:", results_rouge['rougeL'])
    # print("rougeLsum:", results_rouge['rougeLsum'])
    # print("bertscore_f1:", (results_bertscore['f1']))
    # print("bertscore_precision:", np.mean(results_bertscore['precision']))
    # print("bertscore_recall:", np.mean(results_bertscore['recall']))
    # print("wer:", results_wer)
    # print("meteor:", results_meteor['meteor'])
    # print("gleu:", results_gleu['google_bleu'])
    print("parascore:", results_parascore)

    # Write the average results to csv file
    print("Writing results to csv file...")
    writer = csv.writer(csv_file_out, delimiter='\t')
    writer.writerow(['metric', 'result'])
    writer.writerow(['bleu:', results_bleu['bleu']])
    writer.writerow(['rouge1:', results_rouge['rouge1']])
    writer.writerow(['rouge2:', results_rouge['rouge2']])
    writer.writerow(['rougeL:', results_rouge['rougeL']])
    writer.writerow(['rougeLsum:', results_rouge['rougeLsum']])
    writer.writerow(['bertscore_f1:', np.mean(results_bertscore['f1'])])
    writer.writerow(['bertscore_precision:', np.mean(results_bertscore['precision'])])
    writer.writerow(['bertscore_recall:', np.mean(results_bertscore['recall'])])
    writer.writerow(['wer:', results_wer])
    writer.writerow(['meteor:', results_meteor['meteor']])
    writer.writerow(['gleu:', results_gleu['google_bleu']])
    writer.writerow(['parascore:', results_parascore])

    csv_file_in.close()
    csv_file_out.close()


if __name__ == '__main__':
    file_in = ''  # a csv file with the following columns: id, original sentence, length, paraphrase
    file = open(file_in, 'r', encoding='utf8')
    file_out = os.path.join(os.path.dirname(file_in), os.path.basename(file_in).split('.')[0] + '-metrics.' +
                            os.path.basename(file_in).split('.')[-1])
    out = open(file_out, 'w', encoding='utf8', newline='')
    calculate_matrices(file, out)
