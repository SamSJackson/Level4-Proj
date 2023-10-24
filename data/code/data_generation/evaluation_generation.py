import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.code.user_api.Evaluator import Evaluator


delta = 5
cuda = True
model = "gpt2"
documents = 100

evaluation_path = \
    f"../processed/train/model_{model}_{documents}_delta_{delta}_cuda_{cuda}.csv"
gamma_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65]

tokenizer_name = model
df = pd.read_csv(evaluation_path)

evaluator = Evaluator(
    tokenizer_name
)

no_groups = 10
groups_of_df = np.split(df['watermarked'], no_groups)
all_results = []

for i in tqdm.tqdm(range(len(groups_of_df))):
    gamma = gamma_values[i]
    group = groups_of_df[i]
    for text in group:
        results = evaluator.detect(text, delta=delta, gamma=gamma)
        all_results.append(results)

z_scores = [result['z_score'] for result in all_results]
idx = list(range(len(z_scores)))
plt.plot(idx, z_scores, marker='o', linestyle='None')
plt.show()

