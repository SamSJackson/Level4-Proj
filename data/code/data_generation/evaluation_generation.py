import tqdm
import pandas as pd

from transformers import AutoTokenizer
from data.code.implementation.newkirch.extended_watermark_processor import WatermarkDetector
from data.code.implementation.newthickstun.thickstun_detect import permutation_test

base_path = "../../processed/train/paraphrased"
data_path = base_path + "paraphrase_humarin_samples_289_20_11_23.csv"

df = pd.read_csv(data_path)
df = df.dropna()

kthl_watermarked = df["kthl-watermarked"]
kthl_pp_first = df["pp-kthl-first"]

kgw_watermarked = df["kgw-watermarked"]
kgw_pp_first = df["pp-kgw-first"]

z_threshold = 4.0
model_name = "gpt2"
attempt_cuda = True
tokenizer = AutoTokenizer.from_pretrained(model_name)
gamma = 0.25
delta = 10.0

kgw_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                 gamma=gamma,
                                 seeding_scheme="simple_1",
                                 device="cpu",
                                 tokenizer=tokenizer,
                                 z_threshold=4.0,
                                 normalizers=[],
                                 ignore_repeated_ngrams=True)

kgw_wm_pscore = []
kgw_pp_pscore = []

kthl_wm_pscore = []
kthl_pp_pscore = []

kthl_n = 256
kthl_key = 42

for wm_text in tqdm.tqdm(kthl_watermarked):
    tokenized = tokenizer.encode(wm_text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
    p_val = permutation_test(tokenized, kthl_key, kthl_n, len(tokenized), len(tokenizer), n_runs=5)
    kthl_wm_pscore.append(p_val)

for wm_text in tqdm.tqdm(kthl_pp_first):
    tokenized = tokenizer.encode(wm_text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
    p_val = permutation_test(tokenized, kthl_key, kthl_n, len(tokenized), len(tokenizer), n_runs=5)
    kthl_pp_pscore.append(p_val)

for wm_text in tqdm.tqdm(kgw_watermarked):
    score_dict = kgw_detector.detect(wm_text)
    kgw_wm_pscore.append(score_dict["p_value"])

for wm_text in tqdm.tqdm(kgw_pp_first):
    score_dict = kgw_detector.detect(wm_text)
    kgw_pp_pscore.append(score_dict["p_value"])

df["kgw-wm-pscore"] = kgw_wm_pscore
df["kgw-wm-pp-pscore"] = kgw_pp_pscore

df["kthl-wm-pscore"] = kthl_wm_pscore
df["kthl-wm-pp-pscore"] = kthl_pp_pscore

output_path = base_path + f"evaluated/paraphrase_humarin_samples_{len(kgw_wm_pscore)}_EVALUATED_19_11_23.csv"
df.to_csv(output_path, index=False)








