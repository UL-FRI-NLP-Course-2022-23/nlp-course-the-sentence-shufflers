import torch
from parallelbar import progress_map
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import csv

# load the fine-tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "./checkpoints/t5-sl-small-finetuned-MaCoCu-2M(all)-bleu-smallest"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(DEVICE)
model.eval()
model_name = model_checkpoint.split("/")[-1]


def paraphrase(input_text=None):
    if not input_text.startswith("paraphrase: "):
        input_text = "paraphrase: " + input_text

    print(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)

    outputs = model.generate(input_ids,
                             max_length=1000,
                             num_beams=4,
                             # num_beam_groups=6,
                             # no_repeat_ngram_size=2,
                             num_return_sequences=5,
                             do_sample=True,
                             # early_stopping=True
                             )

    # Will create a unique set of output sentences
    sentences = set()
    for gen_sentence in outputs:
        sentences.add(tokenizer.decode(gen_sentence, skip_special_tokens=True))

    for gen_sentence in sentences:
        print("Generated: ", gen_sentence)

    # Try to return different sentence than input
    retval = list(sentences)[0]
    for sentence in list(sentences):
        if sentence != input_text:
            return sentence

    return retval


# Use ctrl + D to finish code
if __name__ == '__main__':
    file = open("paraphrasing-sentences/dataset_for_evaluation.csv", encoding='utf-8', mode='r')
    out_file = open("paraphrasing-sentences/dataset_for_evaluation_paraphrased.csv", encoding='utf-8', mode='w', newline='')
    reader = csv.reader(file, delimiter='\t')
    writer = csv.writer(out_file, delimiter='\t')

    # Use if the csv file has header
    # writer.writerow(next(reader))

    for line in reader:
        output = paraphrase(line[0])
        # line.append(output)
        writer.writerow([line[0], output])
