import tqdm
import pandas as pd

from data.code.user_api.Evaluator import Evaluator

base_path = "../../processed/train/"
data_path = base_path + "paraphrase_humarin_samples_50.csv"

df = pd.read_csv(data_path)
df = df.dropna()

kthl_watermarked = df["kthl-watermarked"]
kthl_pp_first = df["pp-kthl-first"]

kgw_watermarked = df["kgw-watermarked"]
kgw_pp_first = df["pp-kgw-first"]

z_threshold = 4.0
model_name = "gpt2"
attempt_cuda = True
kgw_evaluator = Evaluator(tokenizer_name="gpt2", watermark_name="kirchenbauer", attempt_cuda=attempt_cuda, z_threshold=z_threshold)
kthl_evaluator = Evaluator(tokenizer_name="gpt2",watermark_name="stanford", attempt_cuda=attempt_cuda, z_threshold=z_threshold)

gamma = 0.5
delta = 2.0

kgw_wm_zscore = []
kgw_pp_zscore = []

kthl_wm_zscore = []
kthl_pp_zscore = []

for wm_text in tqdm.tqdm(kthl_watermarked):
    results = kthl_evaluator.detect(wm_text)
    kthl_wm_zscore.append(results["z_score"])

for wm_text in tqdm.tqdm(kthl_pp_first):
    results = kthl_evaluator.detect(wm_text)
    kthl_pp_zscore.append(results["z_score"])

for wm_text in tqdm.tqdm(kgw_watermarked):
    results = kgw_evaluator.detect(wm_text, gamma=gamma, delta=delta)
    kgw_wm_zscore.append(results["z_score"])

for wm_text in tqdm.tqdm(kgw_pp_first):
    results = kgw_evaluator.detect(wm_text, gamma=gamma, delta=delta)
    kgw_pp_zscore.append(results["z_score"])

df["kgw-wm-zscore"] = kgw_wm_zscore
df["kgw-wm-pp-zscore"] = kgw_pp_zscore

df["kthl-wm-zscore"] = kthl_wm_zscore
df["kthl-wm-pp-zscore"] = kthl_pp_zscore


output_path = base_path + f"paraphrase_humarin_samples_50_EVALUATED.csv"
df.to_csv(output_path, index=False)








