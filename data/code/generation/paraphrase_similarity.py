import pandas as pd
import re

from pathlib import Path
from datetime import datetime

from sacremoses import MosesTokenizer
from data.code.implementation.wieting.models import load_model
from data.code.implementation.wieting.measure import make_sim_object, evaluate

def calculate_similarity_and_get_path(
        input_dir: str,
        no_paraphrases: int = 3,
        wieting_path: str = "../implementation/wieting",
        target_dir: str = "../../processed/"
):
    date = datetime.now().strftime("%d_%m_%Y")

    para_model_name = f"{wieting_path}/paraphrase-at-scale-english/model.para.lc.100.pt"
    sp_model = f"{wieting_path}/paraphrase-at-scale-english/paranmt.model"

    para_model, _ = load_model(model_name=para_model_name,
                               sp_model=sp_model,
                               gpu=True)

    entok = MosesTokenizer(lang='en')
    sim_object = make_sim_object(batch_size=32, entok=entok, model=para_model)

    df = pd.read_csv(input_dir)

    created_para_sims_columns = [f"pp-para-sim-{i}" for i in range(1, no_paraphrases+1)]
    df[created_para_sims_columns] = df.apply(lambda row:
                                             evaluate(sim_object, row['kgw-watermarked'], [row[f"pp-kgw-para-{i}"] for i in range(1, no_paraphrases+1)]),
                                             axis='columns',
                                             result_type='expand')

    created_sent_sims_columns = [f"pp-sent-sim-{i}" for i in range(1, no_paraphrases + 1)]
    df[created_sent_sims_columns] = df.apply(lambda row:
                                             evaluate(sim_object, row['kgw-watermarked'], [row[f"pp-kgw-sent-{i}"] for i in range(1, no_paraphrases + 1)]),
                                             axis='columns',
                                             result_type='expand')

    created_word_sims_columns = [f"pp-word-sim-{i}" for i in range(1, no_paraphrases + 1)]
    df[created_word_sims_columns] = df.apply(lambda row:
                                             evaluate(sim_object, row['kgw-watermarked'], [row[f"pp-kgw-word-{i}"] for i in range(1, no_paraphrases + 1)]),
                                             axis='columns',
                                             result_type='expand')

    output_path = Path(f"{target_dir}/similarity/", parents=True, exist_ok=True)
    output_file = f"similarity_{df.shape[0]}_{date}.csv"
    df.to_csv(output_path / output_file, index=False)

    del sp_model
    del para_model

    return output_path / output_file

if __name__ == "__main__":
    calculate_similarity_and_get_path(
        input_dir="../../processed/cleaned_paraphrased/clean_448_03_03_2024.csv",
        target_dir="../../processed/testing/"
    )