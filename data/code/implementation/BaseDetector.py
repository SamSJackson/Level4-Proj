import torch
from tokenizers import Tokenizer


class BaseDetector:

    def __init__(
            self,
            *args,
            device: torch.device = None,
            tokenizer: Tokenizer = None,
            z_threshold: float = 4.0,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

    def detect(self,
               text: str,
               **kwargs
               ) -> dict:
        ## ABSTRACT CLASS
        return {"": "ABSTRACT CLASS"}
