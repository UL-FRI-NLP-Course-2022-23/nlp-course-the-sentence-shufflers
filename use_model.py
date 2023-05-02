from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch

#load the fine tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model_checkpoint = "cjvt/t5-sl-small"
model_checkpoint = "./t5-sl-small-finetuned-MaCoCu-250k"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(DEVICE)

model_name = model_checkpoint.split("/")[-1]

#get output
input_text = "paraphrase: V osnovi moramo paziti predvsem na to, da je gladka in tako tekoča, da jo lahko brez večjih težav vlijemo v ponev in razlijemo po njej."
input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(DEVICE) # move input tensor to the same device as the model
outputs = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2, num_beams=5, num_return_sequences=5)

# print(f"Inutputs: {input_ids}")

# print(f"Outputs: {outputs}")

print("Input:", input_text)

for i in range(5):
    print(f"Output {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}")

#output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#print("Generated:", output_text)

print("Done")
