import torch.cuda
import pandas as pd
import numpy as np
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
        max_length=128
):
    # Problem is that this paraphrase requires some length to be the prompt
    # No prompt does perform worse - albeit still works.
    split_words = text.split()

    # Ensures that the paragraph is 3 times longer than the prompt length at least.
    # If this is not possible, skip prompt
    prompt = "" if len(split_words) < prompt_length*3 else " ".join(split_words[:prompt_length])
    paragraph = " ".join(split_words) if len(prompt) == 0 else " ".join(split_words[prompt_length:])
    input_text = f"lexical = 80, order = 100 {prompt} <sent> {paragraph} </sent>"

    input_ids = tokenizer(
        input_text,
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
    res = f"{prompt} {res[0]}"
    return res

data_path = "../processed/train/wmarked/model_mistralai-Mistral-7B-Instruct-v0_2_100_delta_5_11_01_2024.csv"
wmarked_df = pd.read_csv(data_path)

kgw_documents = wmarked_df["kgw-watermarked"]
print(kgw_documents)
kgw_documents_titled = kgw_documents.apply(lambda x: x.lower().startswith(" title"))
print(kgw_documents_titled)

print(np.sum(kgw_documents_titled) / kgw_documents_titled.shape[0])

# model_path = "finetuning/saved/google-t5-efficient-large-nl32-25_000-finetuned"
# tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-large-nl32")
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map=device)
#
# sample_document = np.random.choice(wmarked_df["kgw-watermarked"])
# prompt_length = 10
#
# paraphrased_response = paraphrase(sample_document, prompt_length, model, tokenizer)
#
# print(f"{'-'*7}\n")
# print(f"{sample_document}\n")
# print(f"{'-'*7}\n")
# print(f"{paraphrased_response}\n")
# print(f"{'-'*7}")