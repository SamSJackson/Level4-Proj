import numpy as np
import pandas as pd
from datetime import datetime
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

def clean_unnecessary_spaces(out_string):
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

train_path = "../../prepared/train/par3/two_trans_train_untokenized.csv"
validation_path = "../../prepared/validation/par3/two_trans_validation_untokenized.csv"

train_df = pd.read_csv(train_path)
eval_df = pd.read_csv(validation_path)

train_df = train_df.sample(frac=1).reset_index(drop=True)
eval_df = eval_df.sample(frac=1).reset_index(drop=True)

train_df = train_df.rename(
    columns={'translation_1': 'input_text', "translation_2": "target_text"}
)

eval_df = eval_df.rename(
    columns={'translation_1': 'input_text', "translation_2": "target_text"}
)

train_df["prefix"] = "paraphrase"
eval_df["prefix"] = "paraphrase"

train_df = train_df[["prefix", "input_text", "target_text"]]
eval_df = eval_df[["prefix", "input_text", "target_text"]]

train_df = train_df.dropna()
eval_df = eval_df.dropna()

train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)

eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

model_args = Seq2SeqArgs()
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_seq_length = 128
model_args.num_train_epochs = 2
model_args.overwrite_output_dir = False
model_args.reprocess_input_data = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.train_batch_size = 8
model_args.use_multiprocessing = False

model_args.do_sample = True
model_args.num_beams = None
model_args.num_return_sequences = 3
model_args.max_length = 128
model_args.top_k = 50
model_args.top_p = 0.95

model_args.wandb_project = "Paragraph-Based Paraphrasing with BART"

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=True,
)

model.train_model(train_df,
                  output_dir="saved/"
                  )
