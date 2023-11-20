import tqdm
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def paraphrase(
        text,
        model,
        tokenizer,
        num_beams=2,
        num_beam_groups=2,
        num_return_sequences=2,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {text}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).to("cpu").input_ids

    outputs = model.generate(
        input_ids,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        max_length=max_length,
        diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
base_path = "../../processed/train/"
file_location = base_path + f"model_gpt2_289_delta_10.0_kgw_kthl.csv"

df = pd.read_csv(file_location)

kthl_watermarked = df["kthl-watermarked"]
kgw_watermarked = df["kgw-watermarked"]

first_kthl_paraphrase = []
first_kgw_paraphrase = []

for text in tqdm.tqdm(kthl_watermarked):
    response = paraphrase(text, model, tokenizer)[0]
    first_kthl_paraphrase.append(response)

for text in tqdm.tqdm(kgw_watermarked):
    response = paraphrase(text, model, tokenizer)[0]
    first_kgw_paraphrase.append(response)

df["pp-kgw-first"] = first_kgw_paraphrase
df["pp-kthl-first"] = first_kthl_paraphrase

output_path = base_path + f"paraphrase_humarin_samples_{len(kgw_watermarked)}_20_11_23.csv"
df.to_csv(output_path, index=False)







