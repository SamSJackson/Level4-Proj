import tqdm
import pandas as pd

import torch
from datetime import datetime
from transformers import AutoTokenizer
from data.code.implementation.maryland.extended_watermark_processor import WatermarkDetector

date = datetime.now().strftime("%d_%m_%Y")
base_path = "../../processed/train/"
data_path = base_path + "cleaned_paraphrased/cleaned_paraphrase_replaced_adjectives_mistralai_51_21_02_2024.csv"

df = pd.read_csv(data_path)
df = df.dropna()

kgw_watermarked = df["kgw-watermarked"]
non_watermarked = df["non-watermarked"]

z_threshold = 4.0
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
attempt_cuda = True
tokenizer = AutoTokenizer.from_pretrained(model_name)
gamma = 0.25
delta = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kgw_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                 gamma=gamma,
                                 seeding_scheme="simple_1",
                                 device=device,
                                 tokenizer=tokenizer,
                                 z_threshold=4.0,
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

for i in range(1, 4):
    kgw_pp = df[f"pp-kgw-{i}"]
    non_pp = df[f"pp-unwatermarked-{i}"]

    df[f"kgw-wm-pp-zscore-{i}"] = paraphrase_scores(kgw_pp)
    df[f"non-wm-pp-zscore-{i}"] = paraphrase_scores(non_pp)

output_path = base_path + f"evaluated/paraphrase_replaced_adjectives_mistralai_{len(kgw_pp)}_EVALUATED_{date}.csv"
df.to_csv(output_path, index=False)








