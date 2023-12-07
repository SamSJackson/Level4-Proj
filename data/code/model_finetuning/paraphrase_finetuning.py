import pandas as pd, numpy as np
import torch, evaluate
from datasets import Dataset
from datetime import datetime
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

date = datetime.now().strftime("%d_%m_%Y")

def preprocess_bart(tokenizer_x, row):
    max_seq_length = 200
    input_ids = tokenizer_x.batch_encode_plus(
        [row["input_text"]],
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )

    target_ids = tokenizer_x.batch_encode_plus(
        [row["target_text"]],
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )

    return {
        "input_ids": input_ids["input_ids"].squeeze(),
        "attention_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }

def compute_metrics(metrics_dict, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {m_name: m.compute(predictions=predictions, references=labels)[m_name] for (m_name, m) in metrics_dict.items()}

metrics = {"accuracy": evaluate.load("accuracy")}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_path = "../../prepared/train/par3/two_trans_train_untokenized.csv"
valid_path = "../../prepared/validation/par3/two_trans_validation_untokenized.csv"

model_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

training_args = TrainingArguments(report_to=None,
                                  output_dir="practice/",
                                  evaluation_strategy="epoch",
                                  learning_rate=1e-5,
                                  num_train_epochs=2,
                                  auto_find_batch_size=True,
                                  save_strategy="epoch")

train_df = pd.read_csv(train_path)
eval_df = pd.read_csv(valid_path)
train_df = train_df.sample(frac=1).reset_index(drop=True)
eval_df = eval_df.sample(frac=1).reset_index(drop=True)

train_df = train_df.rename(
    columns={'translation_1': 'input_text', "translation_2": "target_text"}
)

eval_df = eval_df.rename(
    columns={'translation_1': 'input_text', "translation_2": "target_text"}
)

# Sample while I figure out the errors.
train_dataset = Dataset.from_pandas(train_df.sample(100)).map(lambda x: preprocess_bart(tokenizer, x))
eval_dataset = Dataset.from_pandas(eval_df.sample(20)).map(lambda x: preprocess_bart(tokenizer, x))


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     compute_metrics=lambda x: compute_metrics(metrics, x)
# )
#
# trainer.train()
# print("Finished training")
# trainer.save_model(f"saved/scratch/")
# print("Finished saving")
