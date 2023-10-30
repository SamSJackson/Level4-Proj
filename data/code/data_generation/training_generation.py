import random, tqdm
import pandas as pd

from data.code.user_api.Generator import Generator

training_path = "../../prepared/train/training_untokenized.csv"

sample_size = 50
no_of_documents = 792112
skip = sorted(random.sample(range(1, no_of_documents+1), no_of_documents-sample_size))

df = pd.read_csv(training_path, header=0, skiprows=skip)

print(f"Documents: {sample_size}")

sampled_into_df = df.copy()

z_threshold = 4.0
model_name = "gpt2"
attempt_cuda = True
kgw_generator = Generator(model_name, watermark_name="kirchenbauer", attempt_cuda=attempt_cuda, z_threshold=z_threshold)
kthl_generator = Generator(model_name, watermark_name="stanford", attempt_cuda=attempt_cuda, z_threshold=z_threshold)

kgw_sampled_answers = []
kthl_sampled_answers = []
unwatermarked_sampled_answers = []

gamma = 0.5
delta = 2.0

prompts = df['content-to-sample']

for prompt in tqdm.tqdm(prompts):
    kgw_content = kgw_generator.generate(prompt, gamma=gamma, delta=delta)
    kthl_content = kthl_generator.generate(prompt)
    unwatermarked = kgw_generator.generate(prompt, is_watermark=False) # Content generator does not matter when no watermark

    kgw_sampled_answers.append(kgw_content)
    kthl_sampled_answers.append(kthl_content)
    unwatermarked_sampled_answers.append(unwatermarked)


sampled_into_df["kgw-watermarked"] = kgw_sampled_answers
sampled_into_df["kthl-watermarked"] = kthl_sampled_answers
sampled_into_df["non-watermarked"] = unwatermarked_sampled_answers

output_path = f"../../processed/train/model_{model_name.replace('/', '-')}_{sample_size}_delta_{delta}_cuda_{attempt_cuda}_kgw_kthl.csv"
sampled_into_df.to_csv(output_path, index=False)








