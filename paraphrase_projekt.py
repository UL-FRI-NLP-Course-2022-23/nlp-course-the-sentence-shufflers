import nltk
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import numpy as np
import torch
import evaluate
import transformers

#set this to true if you want to use our pretraind model
RELOAD_MODEL = False 

# ==================== define functions here ====================
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    print("Computing metrics")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# ==================== end of functions ====================
print("Po importih in definicijah")
# nltk.download('punkt')
# print(transformers.__version__)

#chose the model you want to use
if RELOAD_MODEL:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_checkpoint = "TODO: add path to model checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(DEVICE)
else: 
    model_checkpoint = "cjvt/t5-sl-small" 

print(f"Using model {model_checkpoint}")

# Loads our dataset
dataset_name = "MaCoCu-dataset-250k-ids"
raw_datasets = load_dataset('csv', data_files=[f'./MaCoCu-datasets/{dataset_name}.csv'], delimiter='\t', skiprows=1, column_names=['input_ids','input_text','target_text'], split='train')

print(f"Using dataset: {dataset_name}")

# split the dataset
out = raw_datasets.train_test_split(test_size=0.1)
out2 = out['train'].train_test_split(test_size=0.1)

train_set = out2['train']
eval_set = out2['test']
test_set = out['test']


#set up metrics
metric = evaluate.load("bleu") 

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)

# demo of tokenizer
# print("Printing tokenizer")
# with tokenizer.as_target_tokenizer():
#     print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))

# chnage prefix to paraphrase always do this 
prefix = "paraphrase: "

max_input_length = 512
max_target_length = 128

# tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# raw_datasets = raw_datasets.select(list(range(0, 100000)))
tokenized_train_dataset = train_set.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_set.map(preprocess_function, batched=True)
tokenized_test_dataset = test_set.map(preprocess_function, batched=True)


# fine tunning the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-MaCoCu-250k-bleu",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    optim="adamw_torch",
    fp16=False, #prej je bil na True ampak pol je loss skos 0.0
    report_to="none", #api key za wandb
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset= tokenized_train_dataset,
    eval_dataset= tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Zacel sem trenirat")
trainer.train()
print("Koncal sem treniranje")

# save the fine tuned model
trainer.save_model(f"./{model_name}-finetuned-MaCoCu-250k-bleu")

#load the fine tuned model
model = AutoModelForSeq2SeqLM.from_pretrained(f"./{model_name}-finetuned-MaCoCu-250k-bleu")

#get output
input_text = "paraphrase: To je testni primer"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))


print("končal sem")