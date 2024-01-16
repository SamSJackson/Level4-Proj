import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LogitsProcessorList
from data.code.implementation.newkirch.extended_watermark_processor import WatermarkLogitsProcessor

def decode_text(tokenizer, tokens, input_tokens):
    output_tokens = tokens[:, input_tokens["input_ids"].shape[-1]:]
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return output_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token = os.environ['HF_ACCESS_TOKEN']

text_df = pd.read_csv("processed/train/evaluated/paraphrase_humarin_samples_llama-2-7b_230_EVALUATED_05_12_2023.csv")
prompt = text_df['content-to-sample'][0]

model_name = "TheBloke/Llama-2-7b-GPTQ"
config = AutoConfig.from_pretrained(model_name, token=access_token)
config.quantization_config["disable_exllama"] = False
config.quantization_config["exllama_config"] = {"version": 2}
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map=device,
                                             config=config,
                                             token=access_token)

tokenized_text = tokenizer(prompt, return_tensors='pt').to(device)

response = model.generate(**tokenized_text,
                            max_new_tokens=512,
                            do_sample=True,
                            num_beams=3,
                            min_length=3,
                            early_stopping=True,
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.5,
                            logits_processor=[WatermarkLogitsProcessor],
                            pad_token_id=tokenizer.eos_token_id
                          )

response_decoded = tokenizer.batch_decode(response)[0]
print(response_decoded)

response_decoded = decode_text(tokenizer, response, tokenized_text)
print(f"{prompt} //// {response_decoded}")
