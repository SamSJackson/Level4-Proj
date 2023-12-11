import re, tqdm, os
import pandas as pd, numpy as np
from datasets import Dataset

regex_train = re.compile('.*.sents_[1].train_ctrl_ctx*')
regex_valid = re.compile('.*.sents_[1].valid_ctrl_ctx.tsv')
train_directories = [
    root + "/" + file for root, dirs, files in os.walk("../raw/dipper-par3/")
    for file in files if regex_train.match(root + "/" + file)
]
valid_directories = [
    root + "/" + file for root, dirs, files in os.walk("../raw/dipper-par3/")
    for file in files if regex_valid.match(root + "/" + file)
]


train_dataset_dict = {"input_text": [], "target_text": []}
for file_path in tqdm.tqdm(train_directories):
    with open(file_path, encoding='utf8') as file:
        lines = file.readlines()
        train_dataset_dict["input_text"] = train_dataset_dict.get("input_text") + [line.split('\t')[0] for line in lines if len(line.split('\t')) == 2][:25_000]
        train_dataset_dict["target_text"] = train_dataset_dict.get("target_text") + [line.split('\t')[1] for line in lines if len(line.split('\t')) == 2][:25_000]
        lines = None

train_df = pd.DataFrame(train_dataset_dict)
train_df = train_df.replace(to_replace='', value=np.nan).dropna()
train_ds = Dataset.from_pandas(train_df)
train_ds.to_csv("../prepared/train/par3-dipper-25_000/train_combined_sents_1.csv")

train_ds = None
train_dataset_dict = None

valid_dataset_dict = {"input_text": [], "target_text": []}

for file_path in tqdm.tqdm(valid_directories):
    with open(file_path, encoding='utf8') as file:
        lines = file.readlines()
        valid_dataset_dict["input_text"] = valid_dataset_dict.get("input_text") + [line.split('\t')[0] for line in lines if len(line.split('\t')) == 2][:3200]
        valid_dataset_dict["target_text"] = valid_dataset_dict.get("target_text") + [line.split('\t')[1] for line in lines if len(line.split('\t')) == 2][:3200]
        lines = None

valid_df = pd.DataFrame(valid_dataset_dict)
valid_df = valid_df.replace(to_replace='None', value=np.nan).dropna()
valid_ds = Dataset.from_pandas(valid_df)
valid_ds.to_csv("../prepared/validation/par3-dipper-25_000/validation_combined_sents_1.csv")