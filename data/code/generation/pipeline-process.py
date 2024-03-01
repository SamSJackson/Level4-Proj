import tqdm, os
import pandas as pd, numpy as np
from datetime import datetime

import torch.cuda
from sacremoses import MosesTokenizer
from data.code.implementation.wieting.models import load_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    LogitsProcessorList
)
from data.code.implementation.maryland.extended_watermark_processor import (
    WatermarkLogitsProcessor,
    WatermarkDetector
)
from data.code.implementation.wieting.measure import (
    make_sim_object,
    evaluate
)

def cmd_print(text, length=50):
    print(f"{'-' * 50}")
    print(f"{text}")
    print(f"{'-' * 50}")

'''
    Generation Parameters:
    device - string, name of device to run process on ("cuda" or "cpu")
    model_name - string, huggingface name of model to watermark content on
    number_of_documents - int, representing number of watermarked documents to generate
    prompt_training_path - string, path to prompts for watermark generation
    dataframe_size - string, number of rows in prompt dataset
    z_threshold - float, z-score required for watermark classification
    delta - float, logits bias in watermarking process
    gamma - float, fraction of vocabulary to greenlist   
    no_paraphrases - int, number of paraphrases  
'''

'''
    Process:
    1. Start with watermarking documents.
    2. Paraphrase with DIPPER. (Maybe add paraphrase with sentence-based after this)
    3. Clean paraphrases.
    4. Calculate Z-Scores.
    5. Calculate similarity between paraphrases.
'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
prompt_training_path = "../../prepared/train/daigt/daigt_prompts.csv"
number_of_documents = 450
z_threshold = 4.0
dataframe_size = 2421 # DAIGT dataset
delta = 5.0
gamma = 0.25
no_paraphrases = 3

access_token = os.environ['HF_ACCESS_TOKEN']

random_rows = set(np.random.randint(low=1, high=dataframe_size, size=number_of_documents))
skip_numbers = list(set(range(dataframe_size)) - random_rows)

if 0 in skip_numbers:
    skip_numbers.remove(0)
prompt_df = pd.read_csv(prompt_training_path, header=0, skiprows=skip_numbers)

date = datetime.now().strftime("%d_%m_%Y")
print(f"Documents: {number_of_documents}")

clean_model_name = model_name.replace('/', '-').replace('.', '_')
wmark_tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
config = AutoConfig.from_pretrained(model_name, token=access_token)

wmark_model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map=device,
                                             token=access_token,
                                             torch_dtype=torch.bfloat16,
                                             config=config)

kgw_watermark_processor = WatermarkLogitsProcessor(vocab=list(wmark_tokenizer.get_vocab().values()),
                                                   gamma=gamma, delta=delta,
                                                   seeding_scheme="simple_1")
kgw_sampled_answers = []
unwatermarked_sampled_answers = []
tasks = prompt_df["instructions"]

def generate_essay(model_inputs, logitslist=None):
    # Setting `pad_token_id` to `eos_token_id` for open-ended generation.

    # Multinomial Sampling
    if logitslist != None:
        generated_ids = wmark_model.generate(
            model_inputs,
            max_new_tokens=7500,
            do_sample=True,
            pad_token_id=wmark_tokenizer.eos_token_id,
            logits_processor=logitslist
        )
    else:
        generated_ids = wmark_model.generate(
            model_inputs,
            max_new_tokens=7500,
            do_sample=True,
            pad_token_id=wmark_tokenizer.eos_token_id,
        )

    decoded = wmark_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    text = decoded[0].split("[/INST]")[1]
    return text

for chosen_task in tqdm.tqdm(tasks):
    prompt = f'''You are a student working on the following assignment.

    Write an essay based on the following task in no more than a 100 words.
    {chosen_task}
    '''
    messages = [{
        "role": "user",
        "content": prompt
    }]

    model_inputs = wmark_tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    kgw_output_text = generate_essay(model_inputs, LogitsProcessorList([kgw_watermark_processor]))
    kgw_sampled_answers.append(kgw_output_text)

    nwmark_output_text = generate_essay(model_inputs)
    unwatermarked_sampled_answers.append(nwmark_output_text)

prompt_df["kgw-watermarked"] = kgw_sampled_answers
prompt_df["non-watermarked"] = unwatermarked_sampled_answers

watermarked_output_path = f"../../processed/train/wmarked/{clean_model_name}_{prompt_df.shape[0]}_paraphrased_{date}.csv"
prompt_df.to_csv(watermarked_output_path, index=False)

cmd_print("Finished generating documents")
cmd_print(f"No. Documents: {prompt_df.shape[0]} | Paraphrasing documents...")
# Free up space
wmark_model, wmark_tokenizer, prompt_df = None, None, None

def paraphrase(
        p_text,
        p_model,
        p_tokenizer,
        max_length=7500
):
    input_text = f"lexical = 40, order = 0 <sent> {p_text} </sent>"

    p_input_ids = p_tokenizer(
        input_text,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    ).to(device).input_ids

    # As recommended in DIPPER paper.
    p_outputs = p_model.generate(
        p_input_ids,
        top_p=0.75,
        do_sample=True,
        max_new_tokens=7500,
    )

    res = p_tokenizer.batch_decode(p_outputs, skip_special_tokens=True)
    res = f"{' '.join(res)}"
    return res

paraphrase_tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-large-nl32")
paraphrase_model_path = "../finetuning/saved/google-t5-efficient-large-nl32-100_000-finetuned"
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_path, device_map=device)
paraphrase_file_location = watermarked_output_path

paraphrase_df = pd.read_csv(paraphrase_file_location)

kgw_watermarked = paraphrase_df["kgw-watermarked"]
non_watermarked = paraphrase_df["non-watermarked"]

for i in range(3):
    kgw_paraphrase = []
    nwmarked_paraphrased = []

    for text in tqdm.tqdm(kgw_watermarked):
        response = paraphrase(text, paraphrase_model, paraphrase_tokenizer)
        kgw_paraphrase.append(response)

    for text in tqdm.tqdm(non_watermarked):
        response = paraphrase(text, paraphrase_model, paraphrase_tokenizer)
        nwmarked_paraphrased.append(response)

    paraphrase_df[f"pp-kgw-{i + 1}"] = kgw_paraphrase
    paraphrase_df[f"pp-unwatermarked-{i + 1}"] = nwmarked_paraphrased

    kgw_watermarked = kgw_paraphrase.copy()
    non_watermarked = nwmarked_paraphrased.copy()

paraphrase_output_path = f"../../processed/train/paraphrased/{clean_model_name}_{paraphrase_df.shape[0]}_paraphrased_{date}.csv"
paraphrase_df.to_csv(paraphrase_output_path, index=False)

cmd_print("Finished paraphrasing")
cmd_print(f"No. Documents: {paraphrase_df.shape[0]} |Cleaning paraphrase...")
print(paraphrase_df["pp-kgw-1"][0])

# Free up space
paraphrase_model, paraphrase_tokenizer, paraphrase_df = None, None, None


clean_paraphrase_input_path = paraphrase_output_path
clean_paraphrase_df = pd.read_csv(clean_paraphrase_input_path)

kgw_paraphased_column_names = [f"pp-kgw-{i}" for i in range(1,no_paraphrases+1)]
nwm_paraphrased_column_names = [f"pp-unwatermarked-{i}" for i in range(1,no_paraphrases+1)]

base_columns = ["kgw-watermarked", "non-watermarked"]
all_relevant_columns = [*kgw_paraphased_column_names,
                        *nwm_paraphrased_column_names,
                        *base_columns]

conclusion = []
for column in all_relevant_columns:
    conclusion.append(list(clean_paraphrase_df[column].str.split().apply(lambda x: len(x) > 5)))
np_conclusions = np.array(conclusion)

valid_rows = np.prod(np_conclusions, axis=0)
mask = valid_rows % 2 == 0
clean_paraphrase_df = clean_paraphrase_df[~mask]

clean_paraphrase_output_path = f"../../processed/train/cleaned_paraphrased/{clean_model_name}_{clean_paraphrase_df.shape[0]}_clean_paraphrased_{date}.csv"
clean_paraphrase_df.to_csv(clean_paraphrase_output_path, index=False)

cmd_print("Finished cleaning")
cmd_print(f"No. Documents: {clean_paraphrase_df.shape[0]} | Evaluating Z-Scores...")

clean_paraphrase_df = None

evaluation_path = clean_paraphrase_output_path

eval_df = pd.read_csv(evaluation_path)
eval_df = eval_df.dropna()

eval_tokenizer = AutoTokenizer.from_pretrained(model_name)

kgw_watermarked = eval_df["kgw-watermarked"]
non_watermarked = eval_df["non-watermarked"]

kgw_detector = WatermarkDetector(vocab=list(eval_tokenizer.get_vocab().values()),
                                 gamma=gamma,
                                 seeding_scheme="simple_1",
                                 device=device,
                                 tokenizer=eval_tokenizer,
                                 z_threshold=z_threshold,
                                 normalizers=[],
                                 ignore_repeated_ngrams=True)

def paraphrase_scores(l_text):
    array = []
    for wm_text in tqdm.tqdm(l_text):
        score_dict = kgw_detector.detect(wm_text)
        array.append(score_dict["z_score"])
    return array

eval_df["kgw-wm-zscore"] = paraphrase_scores(kgw_watermarked)
eval_df["non-wm-zscore"] = paraphrase_scores(non_watermarked)

for i in range(1, no_paraphrases+1):
    kgw_pp = eval_df[f"pp-kgw-{i}"]
    non_pp = eval_df[f"pp-unwatermarked-{i}"]

    eval_df[f"kgw-wm-pp-zscore-{i}"] = paraphrase_scores(kgw_pp)
    eval_df[f"non-wm-pp-zscore-{i}"] = paraphrase_scores(non_pp)

evaluation_output_path = f"../../processed/train/evaluated/{clean_model_name}_{eval_df.shape[0]}_evaluated_{date}.csv"
eval_df.to_csv(evaluation_output_path, index=False)

eval_tokenizer, kgw_detector = None, None

cmd_print("Finished evaluating")
cmd_print("Calculating paraphrase similarity...")

evaluated_documents_path = evaluation_output_path
wieting_path = "../implementation/wieting"
para_model_name = f"{wieting_path}/paraphrase-at-scale-english/model.para.lc.100.pt"
sp_model = f"{wieting_path}/paraphrase-at-scale-english/paranmt.model"
para_model, _ = load_model(model_name=para_model_name,
                           sp_model=sp_model,
                           gpu=True)

entok = MosesTokenizer(lang='en')
S = make_sim_object(batch_size=32, entok=entok, model=para_model)

similarity_df = pd.read_csv(evaluated_documents_path)
paraphrased_columns = [f"pp-kgw-{i}" for i in range(1, no_paraphrases+1)]
required_columns = ["kgw-watermarked",
                    *paraphrased_columns]

para_df = similarity_df[required_columns]

created_columns = [f"pp-sim-{i}" for i in range(1, no_paraphrases+1)]
# Need to apply
similarity_df[created_columns] = para_df.apply(lambda row: evaluate(S, row['kgw-watermarked'],
                                                                    [row[f"pp-kgw-{i}"] for i in range(1, no_paraphrases+1)]),
                                               axis='columns',
                                               result_type='expand')

output_path = f"../../processed/train/evaluated/{clean_model_name}_{similarity_df.shape[0]}_similarity_evaluated_{date}.csv"
similarity_df.to_csv(output_path, index=False)
cmd_print(f"No. Documents: {similarity_df.shape[0]} | Finished similarity measure")


cmd_print("FINISHED!")


