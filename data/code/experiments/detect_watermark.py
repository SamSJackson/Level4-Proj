import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data.code.implementation.watermark_detector import WatermarkDetector

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.8, # should match original setting
                                        seeding_scheme="simple_1", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        ignore_repeated_bigrams=True
                                       )

with open("outputs/results.txt") as f:
    text = " ".join(f.readlines())

score_dict = watermark_detector.detect(text) # or any other text of interest to analyze
print(score_dict)