import tqdm, os
import pandas as pd, numpy as np
from datetime import datetime

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LogitsProcessorList
from data.code.implementation.maryland.extended_watermark_processor import WatermarkLogitsProcessor

training_path = "../../prepared/train/daigt/daigt_prompts.csv"
access_token = os.environ['HF_ACCESS_TOKEN']

dataframe_size = 2421 # DAIGT dataset
number_of_documents = 50
random_rows = set(np.random.randint(low=1, high=dataframe_size, size=number_of_documents))
skip_numbers = list(set(range(dataframe_size)) - random_rows)

if 0 in skip_numbers:
    skip_numbers.remove(0)
df = pd.read_csv(training_path, header=0, skiprows=skip_numbers)
print(df.columns)

date = datetime.now().strftime("%d_%m_%Y")
print(f"Documents: {number_of_documents}")

z_threshold = 4.0
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

delta = 5.0
gamma = 0.25
clean_model_name = model_name.replace('/', '-').replace('.', '_')
output_path = f"../../processed/train/wmarked/model_{clean_model_name}_{number_of_documents}_delta_{int(delta)}_{date}.csv"
print(f"Output Path: {output_path}")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
config = AutoConfig.from_pretrained(model_name, token=access_token)
# Necessary for GPTQ models
# config.quantization_config["disable_exllama"] = False
# config.quantization_config["exllama_config"] = {"version": 2}
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map=device,
                                             token=access_token,
                                             torch_dtype=torch.bfloat16,
                                             config=config)

kgw_watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=gamma, delta=delta,
                                                   seeding_scheme="simple_1")
kgw_sampled_answers = []
unwatermarked_sampled_answers = []

tasks = df["instructions"]

def generate_essay(model_inputs, logitslist=None):
    # Setting `pad_token_id` to `eos_token_id` for open-ended generation.

    # Multinomial Sampling
    if logitslist != None:
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=7500,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logitslist
        )
    else:
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=7500,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    text = decoded[0].split("[/INST]")[1]
    return text

for chosen_task in tqdm.tqdm(tasks):
    prompt = f'''You are a student working on the following assignment.

    Write an essay based on the following task in no more than a 100 words.
    {chosen_task}
    '''
    messages = [{
        "role": "user",
        "content": prompt
    }]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    kgw_output_text = generate_essay(model_inputs, LogitsProcessorList([kgw_watermark_processor]))
    kgw_sampled_answers.append(kgw_output_text)

    nwmark_output_text = generate_essay(model_inputs)
    unwatermarked_sampled_answers.append(nwmark_output_text)

df["kgw-watermarked"] = kgw_sampled_answers
df["non-watermarked"] = unwatermarked_sampled_answers

df.to_csv(output_path, index=False)

print("Finished generating documents")