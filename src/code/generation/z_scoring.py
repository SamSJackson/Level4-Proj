import tqdm
import pandas as pd
import re

import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer
from src.code.implementation.maryland.extended_watermark_processor import WatermarkDetector

def evaluate_z_scores_and_get_path(
        gamma: float,
        z_threshold: float,
        input_dir: str,
        no_paraphrases: int = 3,
        target_dir: str = "../../processed/"
):

    date = datetime.now().strftime("%d_%m_%Y")

    df = pd.read_csv(input_dir)
    df = df.dropna()

    kgw_watermarked = df["kgw-watermarked"]
    non_watermarked = df["non-watermarked"]

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kgw_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                     gamma=gamma,
                                     seeding_scheme="simple_1",
                                     device=device,
                                     tokenizer=tokenizer,
                                     z_threshold=z_threshold,
                                     normalizers=[],
                                     ignore_repeated_ngrams=True)

    def paraphrase_scores(l_text):
        array = []
        for wm_text in tqdm.tqdm(l_text):
            score_dict = kgw_detector.detect(wm_text)
            array.append(score_dict["z_score"])
        return array

    df["kgw-wm-zscore"] = paraphrase_scores(kgw_watermarked)
    df["non-wm-zscore"] = paraphrase_scores(non_watermarked)

    para_paraphrased = sorted([column for column in df.columns if re.match("pp.*.para-[0-9]", column)])
    sent_paraphrased = sorted([column for column in df.columns if re.match("pp.*.sent-[0-9]", column)])
    perc_word_replaced = sorted([column for column in df.columns if re.match("pp.*.perc.*.word-[0-9]", column)])
    noun_word_replaced = sorted([column for column in df.columns if re.match("pp.*.noun.*.word-[0-9]", column)])

    for i, col in enumerate(para_paraphrased):
        para_pp = df[col]
        score_name = "nowm" if "kgw" not in col else "kgw"
        df[f"{score_name}-para-zscore-{(i+1) % no_paraphrases + 1}"] = paraphrase_scores(para_pp)

    for i, col in enumerate(sent_paraphrased):
        sent_pp = df[col]
        score_name = "nowm" if "kgw" not in col else "kgw"
        df[f"{score_name}-sent-zscore-{(i + 1) % no_paraphrases + 1}"] = paraphrase_scores(sent_pp)

    for i, col in enumerate(perc_word_replaced):
        perc_word_pp = df[col]
        score_name = "nowm" if "kgw" not in col else "kgw"
        df[f"{score_name}-perc-word-zscore-{(i + 1) % no_paraphrases + 1}"] = paraphrase_scores(perc_word_pp)

    for i, col in enumerate(noun_word_replaced):
        noun_word_pp = df[col]
        score_name = "nowm" if "kgw" not in col else "kgw"
        df[f"{score_name}-noun-word-zscore-{(i + 1) % no_paraphrases + 1}"] = paraphrase_scores(noun_word_pp)

    output_path = Path(f"{target_dir}/z_scored/", parents=True, exist_ok=True)
    output_file = f"z_scored_{df.shape[0]}_{date}.csv"
    df.to_csv(output_path / output_file, index=False)

    del tokenizer

    return output_path / output_file


if __name__ == "__main__":
    evaluate_z_scores_and_get_path(
        gamma=0.25,
        z_threshold=4.0,
        input_dir="../../../data/processed/similarity/similarity_497_17_03_2024.csv",
        target_dir="../../../data/processed/"
    )