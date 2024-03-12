import torch

import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LogitsProcessorList

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token = os.environ['HF_ACCESS_TOKEN']
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
config = AutoConfig.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map=device,
                                             token=access_token,
                                             torch_dtype=torch.bfloat16,
                                             config=config)

input_text = "Godzilla could never beat "

model_inputs = tokenizer(input_text, return_tensors='pt').to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=30,
    do_sample=True,
    top_p=0.50,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Prompt + Response: {generated_text}")