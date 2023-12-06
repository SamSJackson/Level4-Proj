import random, tqdm
import pandas as pd, numpy as np
from datetime import datetime

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LogitsProcessorList
from data.code.implementation.newkirch.extended_watermark_processor import WatermarkLogitsProcessor
from data.code.implementation.newthickstun.thickstun_generate import generate_shift

training_path = "../../prepared/train/training_untokenized.csv"

def decode_text(tokenizer, tokens, input_tokens):
    output_tokens = tokens[:, input_tokens["input_ids"].shape[-1]:]
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return output_text

dataframe_size = 559374 # Dataframe Size
number_of_documents = 250
random_rows = set(np.random.randint(low=1, high=dataframe_size, size=number_of_documents))
skip_numbers = list(set(range(dataframe_size)) - random_rows)
skip_numbers.remove(0)
df = pd.read_csv(training_path, header=0, skiprows=skip_numbers)

date = datetime.now().strftime("%d_%m_%Y")
print(f"Documents: {number_of_documents}")

sampled_into_df = df.copy()

z_threshold = 4.0
model_name = "TheBloke/Llama-2-7B-GPTQ"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamma = 0.25
delta = 5.0

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.quantization_config["disable_exllama"] = False
config.quantization_config["exllama_config"] = {"version": 2}

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", config=config)

kgw_watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=gamma, delta=delta,
                                                   seeding_scheme="simple_1")

kgw_sampled_answers = []
kthl_sampled_answers = []
unwatermarked_sampled_answers = []

kthl_m = 80
kthl_n = 256
kthl_key = 42

prompts = sampled_into_df['content-to-sample']

for prompt in tqdm.tqdm(prompts):
    tokenized_input = tokenizer(prompt, return_tensors='pt').to(device)
    kgw_tokens = model.generate(**tokenized_input,
                                max_new_tokens=200,
                                do_sample=True,
                                min_length=10,
                                no_repeat_ngram_size=2,
                                repetition_penalty=1.5,
                                logits_processor=LogitsProcessorList([kgw_watermark_processor]),
                                pad_token_id=tokenizer.eos_token_id)
    kgw_output_text = decode_text(tokenizer, kgw_tokens, tokenized_input)
    kgw_sampled_answers.append(kgw_output_text)

    kthl_tokens = generate_shift(model, tokenized_input["input_ids"], len(tokenizer), kthl_n, kthl_m, kthl_key)[0]
    kthl_text = tokenizer.decode(kthl_tokens, skip_special_tokens=True)

    kthl_sampled_answers.append(kthl_text)

    nwmark_tokens = model.generate(**tokenized_input,
                                   max_new_tokens=200,
                                   do_sample=True,
                                   min_length=10,
                                   no_repeat_ngram_size=2,
                                   repetition_penalty=1.5,
                                   pad_token_id=tokenizer.eos_token_id)
    nwmark_output_text = decode_text(tokenizer, nwmark_tokens, tokenized_input)
    unwatermarked_sampled_answers.append(nwmark_output_text)

sampled_into_df["kgw-watermarked"] = kgw_sampled_answers
sampled_into_df["kthl-watermarked"] = kthl_sampled_answers
sampled_into_df["non-watermarked"] = unwatermarked_sampled_answers

output_path = f"../../processed/train/wmarked/model_{model_name.replace('/', '-')}_{number_of_documents}_delta_{delta}_{date}.csv"
sampled_into_df.to_csv(output_path, index=False)