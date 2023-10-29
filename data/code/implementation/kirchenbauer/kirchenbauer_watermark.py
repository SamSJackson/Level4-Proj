from transformers import LogitsProcessorList

from data.code.implementation.BaseGenerator import BaseGenerator
from data.code.implementation.kirchenbauer.kirchenbauer_logits import KirchenbauerLogitsProcessor


class KirchenbauerProcessor(BaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.model.device
        self.logits_processor = KirchenbauerLogitsProcessor(
            vocab=self.vocab,
            seeding_scheme="simple_1",
            hash_key=self.hash_key,
        )

    def generate(self,
                 prompt,
                 gamma: float = None,
                 delta: float = None,
                 *args) -> str:
        self.logits_processor.set_attributes(gamma, delta)

        tokenized_input = self.tokenizer(prompt, return_tensors='pt')
        tokenized_input = tokenized_input.to(self.device)

        output_tokens = self.model.generate(**tokenized_input,
                                            max_new_tokens=200,
                                            do_sample=True,
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.5,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            logits_processor=LogitsProcessorList([self.watermarker])
                                            )
        return output_tokens
