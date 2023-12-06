import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer

data_path = "../../prepared/train/par3/two_trans_train_untokenized.csv"
dataset = Dataset.from_csv(data_path)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")




