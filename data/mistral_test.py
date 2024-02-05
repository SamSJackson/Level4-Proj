from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from data.code.implementation.newkirch.extended_watermark_processor import WatermarkLogitsProcessor
import torch

device = "cuda"
model_name = "mistralai/Mistral-7B-Instruct-v0.2" # Mistral AI model

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="selfhash")
def generate_essay(prompt):
    messages = [{
        "role": "user",
        "content": prompt
    }]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Setting `pad_token_id` to `eos_token_id` for open-ended generation.
    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=7500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=LogitsProcessorList([watermark_processor])
    )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    text = decoded[0].split("[/INST]")[1]
    return text

prompt = '''You are a student working on the following assignment.

Create an essay based on the following topic in no more than a 100 words.

Topic: What is the meaning of Scott Pilgrim vs. the World? 
'''

text = generate_essay(prompt)
print(text)
