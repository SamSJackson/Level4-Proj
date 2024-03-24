import tqdm, os
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LogitsProcessorList
from data.code.implementation.maryland.extended_watermark_processor import WatermarkLogitsProcessor

def generate_documents_and_get_path(
        gamma: float,
        delta: float,
        no_documents: int,
        target_dir: str = "../processed/",
        input_dir: str = "../prepared/train/daigt/daigt_prompts.csv"
):
    access_token = os.environ['HF_ACCESS_TOKEN']

    df = pd.read_csv(input_dir, header=0).sample(no_documents)

    date = datetime.now().strftime("%d_%m_%Y")
    print(f"Documents: {no_documents}")

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_model_name = model_name.replace('/', '-').replace('.', '_')
    output_path = Path(f"{target_dir}/wmarked/", parents=True, exist_ok=True)
    output_file = f"model_{clean_model_name}_{no_documents}_delta_{int(delta)}_{date}.csv"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    config = AutoConfig.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map=device,
                                                 token=access_token,
                                                 torch_dtype=torch.bfloat16,
                                                 config=config)

    kgw_watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                       gamma=gamma, delta=delta,
                                                       seeding_scheme="simple_1")
    kgw_sampled_answers = []
    unwatermarked_essays = []

    tasks = df["instructions"]
    # unwatermarked_essays = df["text"]

    def generate_essay(model_inputs, logitslist=None):
        # Setting `pad_token_id` to `eos_token_id` for open-ended generation.

        # Multinomial Sampling
        if logitslist != None:
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=7500,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=logitslist
            )
        else:
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=7500,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        text = decoded[0].split("[/INST]")[1]
        return text

    for chosen_task in tqdm.tqdm(tasks):
        prompt = f'''You are a student working on the following assignment.
    
        Write an essay based on the following task in no more than a 100 words:
        {chosen_task}
        '''
        messages = [{
            "role": "user",
            "content": prompt
        }]

        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        kgw_output_text = generate_essay(model_inputs, LogitsProcessorList([kgw_watermark_processor]))
        kgw_sampled_answers.append(kgw_output_text)

        nwmark_output_text = generate_essay(model_inputs)
        unwatermarked_essays.append(nwmark_output_text)

    df["kgw-watermarked"] = kgw_sampled_answers
    df["non-watermarked"] = unwatermarked_essays

    df.drop(columns=["text"])

    df.to_csv(output_path / output_file, index=False)

    del model
    del tokenizer

    return output_path / output_file


if __name__ == "__main__":
    generate_documents_and_get_path(
        gamma=0.25,
        delta=5.0,
        no_documents=5,
        target_dir="../../processed/"
    )
