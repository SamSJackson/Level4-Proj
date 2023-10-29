# import random, tqdm
# import pandas as pd
#
# from data.code.user_api.Generator import Generator
#
# training_path = "../../processed/train/model_gpt2_50_delta_2.0_cuda_True_kgw_kthl.csv"
#
# df = pd.read_csv(training_path)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
def paraphrase(
        question,
        do_sample=True,
        num_beams=2,
        num_beam_groups=2,
        num_return_sequences=2,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).to("cuda").input_ids

    outputs = model.generate(
        input_ids,
        do_sample=False,
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

content = paraphrase("The ongoing state of the war in Ukraine is only impeding progress in the Western progress")
print(content)




