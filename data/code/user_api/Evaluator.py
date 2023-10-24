from transformers import AutoTokenizer
from data.code.implementation.watermark_detector import WatermarkDetector
import torch


class Evaluator:
    def __init__(self,
                 tokenizer_name: str,
                 seeding_scheme: str = "simple_1",
                 attempt_cuda: bool = True
                 ):
        self.seeding_scheme = seeding_scheme

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except OSError as os_error:
            raise Exception(f"{tokenizer_name} is not a valid model identifier or file location")

        self.device = 'cuda' if (attempt_cuda and
                                 torch.cuda.is_available()) else 'cpu'

        self.watermark_detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            seeding_scheme=self.seeding_scheme,
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4,
            ignore_repeated_bigrams=True,
        )

    # Text provided is only the watermarked text without prompt
    def detect(self, text: str, gamma: float, delta: float) -> dict:
        self._set_attributes(gamma, delta)
        return self.watermark_detector.detect(text)

    def _set_attributes(self, gamma: float = None, delta: float = None):
        if gamma:
            self.watermark_detector.set_gamma(gamma)
        if delta:
            self.watermark_detector.set_delta(delta)
