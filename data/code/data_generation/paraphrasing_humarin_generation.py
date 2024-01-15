import torch.cuda
import tqdm
import pandas as pd

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%d_%m_%Y")
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
    ).to(device).input_ids

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

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
base_path = "../../processed/train/"
file_location = base_path + f"wmarked/model_TheBloke-Llama-2-7b-GPTQ_750_delta_5.0_15_12_2023.csv"

df = pd.read_csv(file_location)

kthl_watermarked = df["kthl-watermarked"]
kgw_watermarked = df["kgw-watermarked"]
non_watermarked = df["non-watermarked"]

for i in range(3):
    kthl_paraphrase = []
    kgw_paraphrase = []
    nwmarked_paraphrased = []

    for text in tqdm.tqdm(kthl_watermarked):
        response = paraphrase(text, model, tokenizer)[0]
        kthl_paraphrase.append(response)

    for text in tqdm.tqdm(kgw_watermarked):
        response = paraphrase(text, model, tokenizer)[0]
        kgw_paraphrase.append(response)

    for text in tqdm.tqdm(non_watermarked):
        response = paraphrase(text, model, tokenizer)[0]
        nwmarked_paraphrased.append(response)

    df[f"pp-kthl-{i+1}"] = kthl_paraphrase
    df[f"pp-kgw-{i+1}"] = kgw_paraphrase
    df[f"pp-unwatermarked-{i+1}"] = nwmarked_paraphrased

    kgw_watermarked = kgw_paraphrase.copy()
    non_watermarked = nwmarked_paraphrased.copy()
    kthl_watermarked = kthl_paraphrase.copy()

output_path = base_path + f"paraphrased/paraphrase_humarin_samples_llama-2-7b_{len(kgw_watermarked)}_{date}.csv"
df.to_csv(output_path, index=False)

print("Finished paraphrasing")







