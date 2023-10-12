from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from watermark_generator import WatermarkLogitsProcessor
import torch


class Generator:
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str = None,
                 seeding_scheme: str = "simple_1",
                 attempt_cuda: bool = True
                 ):
        self.seeding_scheme = seeding_scheme
        tokenizer_name = model_name if not tokenizer_name else tokenizer_name

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except OSError as os_error:
            raise Exception(f"{model_name} is not a valid model identifier or file location")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except  OSError as os_error:
            raise Exception(f"{tokenizer_name} is not a valid model identifier or file location")

        self.device = 'cuda' if (attempt_cuda and
                                 torch.cuda.is_available()) else 'cpu'

        self.model = self.model.to(self.device)

        self.watermark_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            seeding_scheme=self.seeding_scheme
        )

    def generate(self,
                 prompt: str,
                 gamma: float = None,
                 delta: float = None,
                 is_watermark: bool = True,
                 max_new_tokens: int = 200,
                 *args) -> str:
        self._set_attributes(gamma, delta)

        tokenized_input = self.tokenizer(prompt, return_tensors='pt')
        tokenized_input = tokenized_input.to(self.device)

        if is_watermark:
            output_tokens = self.model.generate(**tokenized_input,
                                                max_new_tokens=200,
                                                num_beams=1,
                                                do_sample=True,
                                                no_repeat_ngram_size=2,
                                                repetition_penalty=1.5,
                                                logits_processor=LogitsProcessorList([self.watermark_processor])
                                                )
        else:
            output_tokens = self.model.generate(**tokenized_input,
                                                max_new_tokens=200,
                                                do_sample=True,
                                                no_repeat_ngram_size=2,
                                                repetition_penalty=1.5,
                                                )

        output_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        return output_text

    def _set_attributes(self, gamma: float = None, delta: float = None):
        if gamma:
            self.watermark_processor.set_gamma(gamma)
        if delta:
            self.watermark_processor.set_delta(delta)
