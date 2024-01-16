import pandas as pd
import numpy as np
from datetime import datetime

date = datetime.now().strftime("%d_%m_%Y")

base_path = "../../processed/train/"
path = base_path + "paraphrased/paraphrase_dipper_samples_mistralai_51_16_01_2024.csv"
df = pd.read_csv(path)

print(df)
kgw_paraphased_column_names = [f"pp-kgw-{i}" for i in range(1,4)]
nwm_paraphrased_column_names = [f"pp-unwatermarked-{i}" for i in range(1,4)]

base_columns = ["kgw-watermarked", "non-watermarked"]

all_relevant_columns = kgw_paraphased_column_names + nwm_paraphrased_column_names + base_columns

conclusion = []
for column in all_relevant_columns:
    conclusion.append(list(df[column].str.split().apply(lambda x: len(x) > 5)))
np_conclusions = np.array(conclusion)

valid_rows = np.prod(np_conclusions, axis=0)
mask = valid_rows % 2 == 0

df = df[~mask]

output_path = base_path + f"cleaned_paraphrased/cleaned_paraphrase_dipper_samples_mistralai_{df.shape[0]}_{date}.csv"
df.to_csv(output_path, index=False)