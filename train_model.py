import nltk
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    AutoTokenizer
import numpy as np
import torch
import evaluate

max_input_length = 512
max_target_length = 128
batch_size = 16

# set up metrics
metric = evaluate.load("bleu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# nltk.download('punkt')
# print(transformers.__version__)
dataset_name = "MaCoCu-dataset-(all)2M+-filtered-with-ids"


# ==================== define functions here ====================
def preprocess_function(examples):
    # change prefix to paraphrase always do this
    inputs = ["paraphrase: " + doc for doc in examples["input_text"]]
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

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return result


def load_model(model_checkpoint="cjvt/t5-sl-small"):
    if model_checkpoint is not None:
        # Reload from previous checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(DEVICE)
    else:
        # fine-tuning the model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    print(f"Using model {model_checkpoint}")

    return model, tokenizer


def main(tokenizer, model, model_name):
    # Loads our dataset
    raw_datasets = load_dataset('csv', data_files=[f'./dataset/{dataset_name}.csv'], delimiter='\t', skiprows=1,
                                column_names=['input_ids', 'input_text', 'target_text'], split='train')

    print(f"Using dataset: {dataset_name}")

    # split the dataset
    out = raw_datasets.train_test_split(test_size=0.1)
    out2 = out['train'].train_test_split(test_size=0.1)

    train_set = out2['train']
    eval_set = out2['test']
    # test_set = out['test']

    # tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # raw_datasets = raw_datasets.select(list(range(0, 100000)))
    tokenized_train_dataset = train_set.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_set.map(preprocess_function, batched=True)
    # tokenized_test_dataset = test_set.map(preprocess_function, batched=True)

    args = Seq2SeqTrainingArguments(
        model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        optim="adamw_torch",
        fp16=False,  # prej je bil na True, ampak pol je loss skos 0.0
        report_to="none",  # api key za wandb
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Zacel sem trenirat")
    trainer.train()
    print("Koncal sem treniranje")

    save_path = f"./checkpoints/{model_name}-{dataset_name}"
    # save the fine-tuned model
    trainer.save_model(save_path)

    # load the fine-tuned model
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(save_path)

    # get output
    input_text = "paraphrase: Najstniško popivanje in stalno bežanje od doma ni bil niti najmanjši namig?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = trained_model.generate(input_ids,
                                     num_beams=16,
                                     num_beam_groups=4,
                                     do_sample=False,
                                     max_new_tokens=512,
                                     num_return_sequences=6)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("končal sem")


if __name__ == '__main__':
    # Set checkpoint name to reload previous model
    model, tokenizer = load_model()
    main(tokenizer, model, "tss-model")
