import pandas as pd
import torch

from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, documents):
    ppls = []
    for document in tqdm(documents):
        encodings = tokenizer(document, return_tensors='pt').to("cuda")
        max_length = 2048 # OPT sequence length
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean()).item()
        ppls.append(ppl)
    return ppls


def evaluate_perplexity_on_dataframe(input_dir, target_dir, model_name="facebook/opt-2.7b"):
    date = datetime.now().strftime("%d_%m_%Y")

    df = pd.read_csv(input_dir)
    irrelevant_columns = ["id", "instructions", *[col for col in df.columns if "sim" in col or "zscore" in col]]
    relevant_columns = [col for col in df.columns if col not in irrelevant_columns]

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for col in tqdm(relevant_columns):
        ppls = compute_perplexity(model, tokenizer, df[col].tolist())
        df[col + "-ppl"] = ppls

    output_path = Path(f"{target_dir}/perplexity/", parents=True, exist_ok=True)
    output_file = f"perplexity_{df.shape[0]}_{date}.csv"

    df.to_csv(output_path / output_file, index=False)

    return output_path / output_file

if __name__ == "__main__":
    path = "../../processed/z_scored/z_scored_498_18_03_2024.csv"
    evaluate_perplexity_on_dataframe(
        input_dir=path,
        target_dir="../../processed/testing/",
        model_name="facebook/opt-2.7b"
    )
