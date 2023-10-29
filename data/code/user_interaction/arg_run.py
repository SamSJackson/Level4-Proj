import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessorList
from data.code.implementation.kirchenbauer.kirchenbauer_base import WatermarkLogitsProcessor
from data.code.implementation.kirchenbauer.kirchenbauer_detector import WatermarkDetector

'''
- Prompt
- Gamma (portion of vocab is green list)
- Delta (bias towards greenlist in generation)
'''

parser = argparse.ArgumentParser(
                    prog='WATERMARK',
                    description='Generates and detects watermarking')

parser.add_argument('-g',
                    '--gamma',
                    metavar='G',
                    type=float,
                    nargs=1,
                    default=0.5
                    )

parser.add_argument('-d',
                    '--delta',
                    metavar='D',
                    type=float,
                    nargs=1,
                    default=2
                    )

parser.add_argument('--prompt',
                    metavar='X',
                    type=str,
                    nargs='+',
                    required=True
                    )


args = parser.parse_args()
gamma, delta, prompt = args.gamma[0], args.delta[0], " ".join(args.prompt)

print(f"{gamma=}, {delta=}, {prompt=}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model = model.to(device)



watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=gamma,
                                               delta=delta,
                                               seeding_scheme="simple_1"
                                               )

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma, # should match original setting
                                        seeding_scheme="simple_1", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        ignore_repeated_bigrams=True
                                       )

tokenized_input = tokenizer(prompt, return_tensors='pt')
tokenized_input = tokenized_input.to(device)

output_tokens = model.generate(**tokenized_input,
                               max_new_tokens=200,
                               no_repeat_ngram_size=2,
                               logits_processor=LogitsProcessorList([watermark_processor]))

output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
joined_text = " ".join(output_text)


score_dict = watermark_detector.detect(joined_text) # or any other text of interest to analyze
print(score_dict)


