import random, tqdm
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from data.code.implementation.newkirch.extended_watermark_processor import WatermarkLogitsProcessor
from data.code.implementation.newthickstun.thickstun_generate import generate_shift

training_path = "../../prepared/train/training_untokenized.csv"

def decode_text(tokenizer, tokens, input_tokens):
    output_tokens = tokens[:, input_tokens["input_ids"].shape[-1]:]
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return output_text

# sample_size = 200
# no_of_documents = 9233104
# skip = sorted(random.sample(range(1, no_of_documents+1), no_of_documents-sample_size))

df = pd.read_csv(training_path, header=0, skiprows=lambda x: x > 0 and random.random() > 0.0005)

sample_size = df.shape[0]
print(f"Documents: {sample_size}")

sampled_into_df = df.copy()

z_threshold = 4.0
model_name = "gpt2"
attempt_cuda = True

gamma = 0.25
delta = 10.0

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
kgw_watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=gamma,
                                                   delta=delta,
                                                   seeding_scheme="simple_1")


kgw_sampled_answers = []
kthl_sampled_answers = []
unwatermarked_sampled_answers = []

kthl_m = 80
kthl_n = 256
kthl_key = 42

prompts = sampled_into_df['content-to-sample']

for prompt in tqdm.tqdm(prompts):
    tokenized_input = tokenizer(prompt, return_tensors='pt').to("cpu")
    kgw_tokens = model.generate(**tokenized_input,
                                max_new_tokens=200,
                                do_sample=True,
                                no_repeat_ngram_size=2,
                                repetition_penalty=1.5,
                                logits_processor=LogitsProcessorList([kgw_watermark_processor]))
    kgw_output_text = decode_text(tokenizer, kgw_tokens, tokenized_input)
    kgw_sampled_answers.append(kgw_output_text)

    kthl_tokens = generate_shift(model, tokenized_input["input_ids"], len(tokenizer), kthl_n, kthl_m, kthl_key)[0]
    kthl_text = tokenizer.decode(kthl_tokens, skip_special_tokens=True)

    kthl_sampled_answers.append(kthl_text)

    nwmark_tokens = model.generate(**tokenized_input,
                                   max_new_tokens=200,
                                   do_sample=True,
                                   no_repeat_ngram_size=2,
                                   repetition_penalty=1.5)
    nwmark_output_text = decode_text(tokenizer, nwmark_tokens, tokenized_input)
    unwatermarked_sampled_answers.append(nwmark_output_text)

sampled_into_df["kgw-watermarked"] = kgw_sampled_answers
sampled_into_df["kthl-watermarked"] = kthl_sampled_answers
sampled_into_df["non-watermarked"] = unwatermarked_sampled_answers

output_path = f"../../processed/train/model_{model_name.replace('/', '-')}_{sample_size}_delta_{delta}_kgw_kthl.csv"
sampled_into_df.to_csv(output_path, index=False)