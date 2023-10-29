import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.code.static.mersenne import mersenne_rng
from data.code.implementation.BaseGenerator import BaseGenerator


class ThickstunProcessor(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, prompt, seq_len: int = 10, no_tokens: int = 0, *args, **kwargs) -> str:
        seq_len = kwargs.get("seq_len", 10)

        rng = mersenne_rng(self.hash_key)
        xi = torch.tensor([rng.rand() for _ in range(seq_len * self.vocab_size)]).view(seq_len, self.vocab_size)
        shift = torch.randint(seq_len, (1,))

        inputs = prompt.to(self.model.device)
        attn = torch.ones_like(inputs)
        past = None
        for i in range(no_tokens):
            with torch.no_grad():
                if past:
                    output = self.model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.model(inputs)

            probs = torch.nn.functional.softmax(output.logits[:, -1, :self.vocab_size], dim=-1).cpu()
            token = self.exp_sampling(probs, xi[(shift + i) % seq_len, :]).to(self.model.device)
            inputs = torch.cat([inputs, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        return inputs.detach().cpu()

    def exp_sampling(self, probs, u):
        return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
