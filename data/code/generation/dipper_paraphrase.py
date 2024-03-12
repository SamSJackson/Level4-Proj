import torch.cuda
import tqdm
import pandas as pd

from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def dipper_paraphrase_and_get_path(
        lexical: int,
        order: int,
        input_dir: str,
        no_paraphrases: int = 3,
        target_dir: str = "../../processed/",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date = datetime.now().strftime("%d_%m_%Y")

    def paraphrase(
            text,
            model,
            tokenizer,
            lexical: int,
            order: int,
            max_length=7500
    ):
        input_text = f"lexical = {lexical}, order = {order} <sent> {text} </sent>"

        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ).to(device).input_ids

        # As recommended in DIPPER paper.
        outputs = model.generate(
            input_ids,
            top_p=0.75,
            do_sample=True,
            max_new_tokens=7500,
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        res = f"{' '.join(res)}"
        return res

    tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-large-nl32")
    model_path = "../finetuning/saved/google-t5-efficient-large-nl32-finetuned"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map=device)

    df = pd.read_csv(input_dir)

    kgw_watermarked = df["kgw-watermarked"]
    non_watermarked = df["non-watermarked"]

    for i in range(no_paraphrases):
        kgw_paraphrase = []
        nwmarked_paraphrased = []

        for text in tqdm.tqdm(kgw_watermarked):
            response = paraphrase(text, model, tokenizer, lexical, order)
            kgw_paraphrase.append(response)

        for text in tqdm.tqdm(non_watermarked):
            response = paraphrase(text, model, tokenizer, lexical, order)
            nwmarked_paraphrased.append(response)

        df[f"pp-kgw-para-{i+1}"] = kgw_paraphrase
        df[f"pp-unwatermarked-para-{i+1}"] = nwmarked_paraphrased

        kgw_watermarked = kgw_paraphrase.copy()
        non_watermarked = nwmarked_paraphrased.copy()

    output_path = Path(f"{target_dir}/paraphrased/", parents=True, exist_ok=True)
    output_file = f"dipper_paraphrased_{df.shape[0]}_{date}.csv"
    df.to_csv(output_path / output_file, index=False)

    del model
    del tokenizer

    return output_path / output_file

if __name__ == "__main__":
    dipper_paraphrase_and_get_path(
        lexical=40,
        order=0,
        input_dir="../../processed/wmarked/mistralai-Mistral-7B-Instruct-v0_2_4_paraphrased_01_03_2024.csv",
        target_dir="../../processed/",
    )