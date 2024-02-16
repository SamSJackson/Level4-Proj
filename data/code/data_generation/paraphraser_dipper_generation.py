import torch.cuda
import tqdm
import pandas as pd

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%d_%m_%Y")
def paraphrase(
        text,
        prompt_length,
        model,
        tokenizer,
        num_beams=2,
        num_beam_groups=2,
        num_return_sequences=2,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        max_length=7500
):
    # Problem is that this paraphrase requires some length to be the prompt
    # No prompt does perform worse - albeit still works.
    split_words = text.split()

    # Ensures that the paragraph is X times longer than the prompt length at least.
    # If this is not possible, skip prompt
    prompt = "" if len(split_words) < prompt_length*2 else " ".join(split_words[:prompt_length])
    paragraph = " ".join(split_words) if len(prompt) == 0 else " ".join(split_words[prompt_length:])
    input_text = f"lexical = 40, order = 0 {prompt} <sent> {paragraph} </sent>"

    input_ids = tokenizer(
        input_text,
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).to(device).input_ids

    # As recommended in DIPPER paper.
    outputs = model.generate(
        input_ids,
        top_p=0.75,
        do_sample=True,
    )

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
    res = f"{prompt} {' '.join(res)}"
    return res

tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-large-nl32")
model_path = "../model_finetuning/saved/google-t5-efficient-large-nl32-25_000-finetuned"
base_path = "../../processed/train/"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="cuda")
file_location = base_path + f"wmarked/model_mistralai-Mistral-7B-Instruct-v0_2_50_delta_5_16_01_2024.csv"

df = pd.read_csv(file_location)
prompt_length = 7

kgw_watermarked = df["kgw-watermarked"]
non_watermarked = df["non-watermarked"]

for i in range(3):
    kgw_paraphrase = []
    nwmarked_paraphrased = []

    for text in tqdm.tqdm(kgw_watermarked):
        response = paraphrase(text, prompt_length, model, tokenizer)
        kgw_paraphrase.append(response)

    for text in tqdm.tqdm(non_watermarked):
        response = paraphrase(text, prompt_length, model, tokenizer)
        nwmarked_paraphrased.append(response)

    df[f"pp-kgw-{i+1}"] = kgw_paraphrase
    df[f"pp-unwatermarked-{i+1}"] = nwmarked_paraphrased

    kgw_watermarked = kgw_paraphrase.copy()
    non_watermarked = nwmarked_paraphrased.copy()

output_path = base_path + f"paraphrased/paraphrase_dipper_samples_mistralai_{len(kgw_watermarked)}_{date}.csv"
df.to_csv(output_path, index=False)

print("Finished paraphrasing")







