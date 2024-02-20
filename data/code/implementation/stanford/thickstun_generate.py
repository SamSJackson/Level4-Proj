import os, argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.code.implementation.stanford.mersenne import mersenne_rng

def generate_shift(model,prompt,vocab_size,n,m,key):
    rng = mersenne_rng(key)
    xi = torch.tensor([rng.rand() for _ in range(n*vocab_size)]).view(n,vocab_size)
    shift = torch.randint(n, (1,))

    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1, :vocab_size], dim=-1).cpu()
        token = exp_sampling(probs,xi[(shift+i)%n,:]).to(model.device)
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

def exp_sampling(probs,u):
    return torch.argmax(u ** (1/probs),axis=1).unsqueeze(-1)

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    tokens = tokenizer.encode(args.prompt, return_tensors='pt', truncation=True, max_length=2048)

    watermarked_tokens = generate_shift(model,tokens,len(tokenizer),args.n,args.m,args.key)[0]
    watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

    print(watermarked_text)