import random, tqdm
import numpy as np
import pandas as pd

from Generator import Generator

training_path = "../prepared/train/training_untokenized.csv"
p = 0.003 / 100

# Reads in approximately one percent of all values - only taking row when random number < p
df = pd.read_csv(training_path, header=0, skiprows=lambda i: i > 0 and random.random() > p)

no_documents = df.shape[0]
print(f"Documents: {no_documents}")

sampled_into_df = df.copy()

model_name = "gpt2"
attempt_cuda = True
generator = Generator(model_name, attempt_cuda=attempt_cuda)

no_groups = 5
groups_of_df = np.array_split(df['content-to-sample'], no_groups)
watermarked_sampled_answers = []
normal_sampled_answers = []

for group in tqdm.tqdm(groups_of_df):
    for prompt in group:
        content = generator.generate(prompt, gamma=0.2, delta=5)
        non_watermarked = generator.generate(prompt, is_watermark=False)

        watermarked_sampled_answers.append(content)
        normal_sampled_answers.append(non_watermarked)

sampled_into_df["watermarked"] = watermarked_sampled_answers
sampled_into_df["non-watermarked"] = normal_sampled_answers

sampled_into_df.to_csv(f"../processed/train/example_{p*100}_{no_documents}_cuda_{attempt_cuda}.csv")








