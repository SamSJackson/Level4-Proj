import re, tqdm, os
import pandas as pd, numpy as np
from datasets import Dataset
import random

# regex_train = re.compile('.*.sents_.*.train_ctrl_no_ctx*')
# regex_valid = re.compile('.*.sents_.*.valid_ctrl_no_ctx.tsv')

regex_train = re.compile('.*.sents_.*.train_ctrl_no_ctx*')
regex_valid = re.compile('.*.sents_.*.valid_ctrl_no_ctx.tsv')


train_directories = [
    root + "/" + file for root, dirs, files in os.walk("../raw/dipper-par3/")
    for file in files if regex_train.match(root + "/" + file)
]
valid_directories = [
    root + "/" + file for root, dirs, files in os.walk("../raw/dipper-par3/")
    for file in files if regex_valid.match(root + "/" + file)
]

def make_dataset(directories, sample_amount):
    dataset_dict = {"input_text": [], "target_text": []}
    for file_path in tqdm.tqdm(directories):
        with open(file_path, encoding='utf8') as f:
            lines = f.readlines()
            parsed_lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2]
            random_parsed_lines = random.sample(parsed_lines, sample_amount)
            dataset_dict["input_text"] = dataset_dict.get("input_text") + [p[0] for p in random_parsed_lines]
            dataset_dict["target_text"] = dataset_dict.get("target_text") + [p[1] for p in random_parsed_lines]
            lines = None
    df = pd.DataFrame(dataset_dict)
    df = df.replace(to_replace='', value=np.nan).dropna()
    return Dataset.from_pandas(df)

aim = 100_000
validation_fraction = 0.2

train_sample_amount = int(np.ceil(aim / len(train_directories)))
validation_sample_amount = int(train_sample_amount * validation_fraction)

train_ds = make_dataset(train_directories, train_sample_amount)
print(train_ds)
train_ds.to_csv("../prepared/train/par3-dipper-100_000/train_sampled.csv")

valid_ds = make_dataset(valid_directories, validation_sample_amount)
print(valid_ds)
valid_ds.to_csv("../prepared/validation/par3-dipper-20_000/validation_sampled.csv")