import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# load the fine-tuned model
# DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "./checkpoints/t5-sl-small-finetuned-MaCoCu-2M(all)-bleu-smallest"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(DEVICE)
model.eval()
model_name = model_checkpoint.split("/")[-1]


def paraphrase(input_text=None):
    if input_text is None:
        # Get input text
        input_text = input("paraphrase: ").strip()
        input_text = "paraphrase: " + input_text

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


# Use ctrl + D to finish code
if __name__ == '__main__':
    while True:
        paraphrase()
