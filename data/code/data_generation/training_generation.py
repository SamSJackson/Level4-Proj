import random, tqdm
import numpy as np
import pandas as pd

from data.code.user_api.Generator import Generator
import os

training_path = "../../prepared/train/training_untokenized.csv"
sample_size = 10
no_of_documents = 792112
skip = sorted(random.sample(range(1, no_of_documents+1), no_of_documents-sample_size))

df = pd.read_csv(training_path, header=0, skiprows=skip)

print(f"Documents: {sample_size}")

sampled_into_df = df.copy()

model_name = "gpt2"
attempt_cuda = True
generator = Generator(model_name, attempt_cuda=attempt_cuda)

no_groups = 10
groups_of_df = np.split(df['content-to-sample'], no_groups)
gamma_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65]
watermarked_sampled_answers = []
normal_sampled_answers = []

delta_fixed = 5
for i in tqdm.tqdm(range(len(groups_of_df))):
    gamma = gamma_values[i]
    group = groups_of_df[i]
    for prompt in group:
        content = generator.generate(prompt, gamma=gamma, delta=delta_fixed)
        non_watermarked = generator.generate(prompt, is_watermark=False)

        watermarked_sampled_answers.append(content)
        normal_sampled_answers.append(non_watermarked)

sampled_into_df["watermarked"] = watermarked_sampled_answers
sampled_into_df["non-watermarked"] = normal_sampled_answers

sampled_into_df.to_csv(
    f"../processed/train/model_{model_name.replace('/', '-')}_{sample_size}_delta_{delta_fixed}_cuda_{attempt_cuda}.csv")








