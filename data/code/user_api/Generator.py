import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from data.code.implementation.BaseGenerator import BaseGenerator

from data.code.implementation.kirchenbauer.kirchenbauer_watermark import KirchenbauerProcessor
from data.code.implementation.thickstun.jthickstun_watermark import ThickstunProcessor

WATERMARKS = {"stanford": ThickstunProcessor,
              "kirchenbauer": KirchenbauerProcessor,
              }


class Generator(BaseGenerator):
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str = None,
                 watermark_name: str = "kirchenbauer",
                 seeding_scheme: str = "simple_1",
                 attempt_cuda: bool = True,
                 *args,
                 **kwargs,
                 ):
        self.seeding_scheme = seeding_scheme
        tokenizer_name = model_name if not tokenizer_name else tokenizer_name

        self.delta = kwargs.get('delta')
        self.gamma = kwargs.get('gamma')
        self.hash_key = kwargs.get('hash_key', 15485863)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except OSError:
            raise Exception(f"{model_name} is not a valid model identifier or file location")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except OSError:
            raise Exception(f"{tokenizer_name} is not a valid tokenizer identifier or file location")

        try:
            assert watermark_name.lower() in WATERMARKS
            self.watermark_name = watermark_name.lower()
        except:
            raise Exception(f"{watermark_name} is not a valid watermark\nPick from: {WATERMARKS}")

        self.device = 'cuda' if (attempt_cuda and
                                 torch.cuda.is_available()) else 'cpu'

        self.model = self.model.to(self.device)
        self.vocab = list(self.tokenizer.get_vocab().values())

        super().__init__(self.vocab, self.model, self.tokenizer, self.hash_key)

        self.watermarker = WATERMARKS[self.watermark_name](
            vocab=self.vocab,
            model=self.model,
            tokenizer=self.tokenizer,
            hash_key=self.hash_key,
            *args,
            **kwargs,
        )

    def generate(self,
                 prompt,
                 *args,
                 **kwargs) -> str:
        tokenized_input = self.tokenizer(prompt, return_tensors='pt')
        tokenized_input = tokenized_input.to(self.device)
        is_watermark = kwargs.get('is_watermark', False)

        if is_watermark:
            output_tokens = self.watermarker.generate(tokenized_input, args, kwargs)
        else:
            output_tokens = self.model.generate(**tokenized_input,
                                                max_new_tokens=200,
                                                do_sample=True,
                                                no_repeat_ngram_size=2,
                                                repetition_penalty=1.5,
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                )
        output_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]  # Returns as list[str]
        return output_text
