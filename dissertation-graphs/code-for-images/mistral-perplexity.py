import pandas as pd
from evaluate import load

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
perplexity = load("perplexity", module_type="metric")

df = pd.read_csv("../../data/processed/z_scored/evaluated_497_10_03_2024.csv")

wmarked = df["kgw-watermarked"]
pp_para = df["pp-kgw-para-1"]
pp_para_2 = df["pp-kgw-para-2"]
pp_para_3 = df["pp-kgw-para-3"]

index = 0
perp_og = perplexity.compute(predictions=[wmarked[index], pp_para[index], pp_para_2[index], pp_para_3[index]], model_id=model_name)
print(f"Perplexity: {perp_og['perplexities']}")