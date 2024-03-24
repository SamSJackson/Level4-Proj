import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

def clean_paraphrases_and_get_path(
        input_dir: str,
        target_dir: str,
):
    date = datetime.now().strftime("%d_%m_%Y")

    df = pd.read_csv(input_dir)

    relevant_columns = [col for col in df.columns if re.match(".*pp*.", col)]
    base_columns = ["kgw-watermarked", "non-watermarked"]

    all_relevant_columns = relevant_columns + base_columns

    conclusion = []
    for column in all_relevant_columns:
        conclusion.append(list(df[column].str.split().apply(lambda x: len(x) > 5)))
    np_conclusions = np.array(conclusion)

    valid_rows = np.prod(np_conclusions, axis=0)
    mask = valid_rows % 2 == 0

    df = df[~mask]

    output_path = Path(f"{target_dir}/cleaned_paraphrased/", parents=True, exist_ok=True)
    output_file = f"clean_{df.shape[0]}_{date}.csv"
    df.to_csv(output_path / output_file, index=False)

    return output_path / output_file

if __name__ == "__main__":
    clean_paraphrases_and_get_path(
        input_dir="../../../data/processed/paraphrased/noun_word_replaced_500_17_03_2024.csv",
        target_dir="../../../data/processed/"
    )