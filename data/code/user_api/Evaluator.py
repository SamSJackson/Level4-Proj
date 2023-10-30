import torch
from transformers import AutoTokenizer

from data.code.implementation.kirchenbauer.kirchenbauer_detector import KirchenbauerDetector
from data.code.implementation.thickstun.jthickstun_detect import ThickstunDetector

EVALUATORS = {
    "stanford": ThickstunDetector,
    "kirchenbauer": KirchenbauerDetector,
}

class Evaluator:
    def __init__(self,
                 *args,
                 tokenizer_name: str,
                 watermark_name: str = "kirchenbauer",
                 seeding_scheme: str = "simple_1",
                 attempt_cuda: bool = True,
                 **kwargs
                 ):
        self.seeding_scheme = seeding_scheme
        self.z_threshold = kwargs.get("z_threshold", 4.0)
        kwargs.pop("z_threshold")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.vocab = list(self.tokenizer.get_vocab().values())
        except OSError:
            raise Exception(f"{tokenizer_name} is not a valid model identifier or file location")

        try:
            assert watermark_name.lower() in EVALUATORS.keys()
            self.evaluator_name = watermark_name.lower()
        except:
            raise Exception(f"{watermark_name} is not a valid watermark\nPick from: {EVALUATORS.keys()}")

        self.device = 'cuda' if (attempt_cuda and
                                 torch.cuda.is_available()) else 'cpu'

        self.evaluator = EVALUATORS[self.evaluator_name](
            *args,
            vocab=self.vocab,
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=self.z_threshold,
            **kwargs
        )

    # Text provided is only the watermarked text without prompt
    def detect(self, text: str, *args, **kwargs) -> dict:
        return self.evaluator.detect(text=text, *args, **kwargs)
