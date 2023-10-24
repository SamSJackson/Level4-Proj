import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessorList
from data.code.implementation.watermark_generator import WatermarkLogitsProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

model = model.to(device)

input_text = "I am often wondering about the significance of life without "

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.8,
                                               delta=3.0,
                                               seeding_scheme="simple_1") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.

tokenized_input = tokenizer(input_text, return_tensors='pt')
tokenized_input = tokenized_input.to(device)

# Greedy Generation
output_tokens = model.generate(**tokenized_input,
                               max_new_tokens=200,
                               no_repeat_ngram_size=2,
                               logits_processor=LogitsProcessorList([watermark_processor]))

# Beam Generation
# output_tokens = model.generate(**tokenized_input,
#                                max_new_tokens=30,
#                                num_return_sequences=5,
#                                no_repeat_ngram_size=2,
#                                repetition_penalty=1.5,
#                                top_p=0.92,
#                                temperature=0.85,
#                                do_sample=True,
#                                top_k=50,
#                                early_stopping=True,
#                                logits_processor=LogitsProcessorList([watermark_processor]))

# # if decoder only model, then we need to isolate the
# # newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

# for response in output_text:
#     print(f"{response} - END")

with open("outputs/results.txt", 'w', encoding='utf-8') as f:
    if len(output_text) == 1:
        f.write("\n"+output_text[0])

