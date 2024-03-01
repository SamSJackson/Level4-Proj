import pandas as pd, numpy as np
import torch, evaluate
from datasets import Dataset, load_dataset
from datetime import datetime
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

date = datetime.now().strftime("%d_%m_%Y")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_path = "../../prepared/train/par3-dipper-100_000/train_sampled.csv"
valid_path = "../../prepared/validation/par3-dipper-20_000/validation_sampled.csv"

model_name = "google/t5-efficient-large-nl32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loaded tokenizer")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda")
print("Loaded model")

def preprocess_function(row):
    try:
        max_seq_length = 512
        input_ids = tokenizer.batch_encode_plus(
            [row["input_text"]],
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        target_ids = tokenizer.batch_encode_plus(
            [row["target_text"]],
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    except:
        print(f"{row['input_text'], row['target_text']}")

    return {
        "input_ids": input_ids["input_ids"].squeeze(),
        "attention_mask": input_ids["attention_mask"].squeeze(),
        "labels": target_ids["input_ids"].squeeze(),
    }

metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions_text = tokenizer.decode(logits, skip_special_tokens=True)
    labels_text = tokenizer.decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=predictions_text, references=labels_text)


training_args = Seq2SeqTrainingArguments(report_to="none",
                                         output_dir="practice/",
                                         evaluation_strategy="no",# Set to epoch after compute_metrics problem fixed.
                                         learning_rate=1e-4,
                                         gradient_checkpointing=True,
                                         bf16=True,
                                         num_train_epochs=2,
                                         auto_find_batch_size=True,
                                         generation_num_beams=2,
                                         generation_max_length=200,
                                         save_strategy="epoch")


train_dataset = load_dataset('csv', data_files=train_path)['train'].map(
    preprocess_function,
    batched=False
)

# Claims 'train' but this is actually the validation dataset.
# Could not save dataset with only validation.
eval_dataset = load_dataset('csv', data_files=valid_path)['train'].map(
    preprocess_function,
    batched=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
print("Finished training")
model_path = model_name.replace("/", "-")
trainer.save_model(f"saved/{model_path}-100_000-finetuned")
print("Finished saving")
