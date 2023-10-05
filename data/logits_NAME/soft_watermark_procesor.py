import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor

class WatermarkBase:
    def __init__(
            self,
            vocab: list[int] = None
    ):
        pass