import torch.cuda
import tqdm
import pandas as pd

from datetime import datetime
from pathlib import Path
from nltk import tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def sentence_paraphrase_and_get_path(
        input_dir: str,
        no_paraphrases: int = 3,
        target_dir: str = "../../processed/"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date = datetime.now().strftime("%d_%m_%Y")
    def sentence_paraphrase(
            text,
            model,
            tokenizer,
            max_length=7500
    ):
        sent_split_text = tokenize.sent_tokenize(text)
        sent_paraphrased = []
        for sentence in sent_split_text:
            prompt = "paraphrase:"
            input_ids = tokenizer(
                f'{prompt} {sentence.strip()}',
                return_tensors="pt", padding="longest",
                max_length=max_length,
                truncation=True,
            ).to(device).input_ids

            outputs = model.generate(
                input_ids,
                top_p=0.75,
                do_sample=True,
                max_new_tokens=7500,
            )

            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sent_paraphrased.append(res[0])

        paraphrased = ". ".join(sent_paraphrased)
        return paraphrased

    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
    df = pd.read_csv(input_dir)

    kgw_watermarked = df["kgw-watermarked"]
    non_watermarked = df["non-watermarked"]

    for i in range(no_paraphrases):
        kgw_paraphrase = []
        nwmarked_paraphrased = []

        for text in tqdm.tqdm(kgw_watermarked):
            response = sentence_paraphrase(text, model, tokenizer)
            kgw_paraphrase.append(response)

        for text in tqdm.tqdm(non_watermarked):
            response = sentence_paraphrase(text, model, tokenizer)
            nwmarked_paraphrased.append(response)

        df[f"pp-kgw-sent-{i+1}"] = kgw_paraphrase
        df[f"pp-unwatermarked-sent-{i+1}"] = nwmarked_paraphrased

        kgw_watermarked = kgw_paraphrase.copy()
        non_watermarked = nwmarked_paraphrased.copy()

    output_path = Path(f"{target_dir}/paraphrased/", parents=True, exist_ok=True)
    output_file = f"sentence_paraphrased_{df.shape[0]}_{date}.csv"
    df.to_csv(output_path / output_file, index=False)

    del model
    del tokenizer

    return output_path / output_file

if __name__ == "__main__":
    sentence_paraphrase_and_get_path(
        input_dir="../../processed/wmarked/mistralai-Mistral-7B-Instruct-v0_2_4_paraphrased_01_03_2024.csv",
        target_dir="../../processed/",
    )









