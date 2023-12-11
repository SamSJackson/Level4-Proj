import pandas as pd

data_path = "../processed/train/evaluated/paraphrase_humarin_samples_llama-2-7b_230_EVALUATED_05_12_2023.csv"

df = pd.read_csv(data_path)

prompt = df['content-to-sample']
actual_response = df['content-actual-ending']
wmarked = df['kgw-watermarked']
one_paraphrase = df['pp-kgw-1']

def pretty_print(index):
    c_prompt = prompt[index]
    c_response = actual_response[index]
    c_wmarked = wmarked[index]
    c_pp = one_paraphrase[index]

    print(f"{c_prompt}\n\n")
    print(f"Actual: {c_response}\n\n")
    print(f"Watermarked: {c_wmarked}\n\n")
    print(f"Paraphrased: {c_pp}")
    return
pretty_print(4)