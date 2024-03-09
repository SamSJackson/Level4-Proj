import pandas as pd
import re

from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer, util

def evaluate(model, source : str, paraphrases : list[str]):
    source_embedding = model.encode(source)
    paraphrase_embeddings = [model.encode(para) for para in paraphrases]
    similarities = [util.cos_sim(source_embedding, par_embed).item() for par_embed in paraphrase_embeddings]
    return similarities

def calculate_similarity_and_get_path(
        input_dir: str,
        no_paraphrases: int = 3,
        target_dir: str = "../../processed/"
):
    date = datetime.now().strftime("%d_%m_%Y")

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    df = pd.read_csv(input_dir)

    created_para_sims_columns = [f"pp-para-sim-{i}" for i in range(1, no_paraphrases+1)]
    df[created_para_sims_columns] = df.apply(lambda row:
                                             evaluate(model, row['kgw-watermarked'], [row[f"pp-kgw-para-{i}"] for i in range(1, no_paraphrases+1)]),
                                             axis='columns',
                                             result_type='expand')

    created_sent_sims_columns = [f"pp-sent-sim-{i}" for i in range(1, no_paraphrases + 1)]
    df[created_sent_sims_columns] = df.apply(lambda row:
                                             evaluate(model, row['kgw-watermarked'], [row[f"pp-kgw-sent-{i}"] for i in range(1, no_paraphrases + 1)]),
                                             axis='columns',
                                             result_type='expand')

    created_word_sims_columns = [f"pp-word-sim-{i}" for i in range(1, no_paraphrases + 1)]
    df[created_word_sims_columns] = df.apply(lambda row:
                                             evaluate(model, row['kgw-watermarked'], [row[f"pp-kgw-word-{i}"] for i in range(1, no_paraphrases + 1)]),
                                             axis='columns',
                                             result_type='expand')

    output_path = Path(f"{target_dir}/similarity/", parents=True, exist_ok=True)
    output_file = f"similarity_st_{df.shape[0]}_{date}.csv"
    df.to_csv(output_path / output_file, index=False)

    return output_path / output_file

if __name__ == "__main__":
    calculate_similarity_and_get_path(
        input_dir="../../processed/cleaned_paraphrased/clean_448_03_03_2024.csv",
        target_dir="../../processed/testing/"
    )