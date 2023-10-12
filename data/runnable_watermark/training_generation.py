import random
import numpy as np
import pandas as pd

from Generator import Generator

training_path = "../prepared/train/training_untokenized.csv"
p = 0.005 / 100

# Reads in approximately one percent of all values - only taking row when random number < p
df = pd.read_csv(training_path, header=0, skiprows=lambda i: i > 0 and random.random() > p)

model_name = "gpt2"
generator = Generator(model_name, attempt_cuda=False)

no_groups = 5
groups_of_df = np.array_split(df['content-to-sample'], no_groups)

first_item = str(groups_of_df[0][0])
content = generator.generate(first_item, gamma=0.2, delta=5)
non_watermarked = generator.generate(first_item, is_watermark=False)

print(f"{first_item}(WATERMARKED GENERATED CONTENT){content[0]}(END)")
print(f"{first_item}(NON-WATERMARKED GENERATED CONTENT){non_watermarked[0]}(END)")







