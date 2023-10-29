import numpy as np

from data.code.static.mersenne import mersenne_rng
from data.code.static.levenshtein import levenshtein

from data.code.implementation.BaseDetector import BaseDetector

class ThickstunDetector(BaseDetector):

    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.vocab = list(self.tokenizer.get_vocab().values())
        self.vocab_size = len(self.vocab)

    def detect(self,
               text: str,
               hash_key: int = 15485863,
               seq_len: int = 0,
               n_runs: int = 100,
               **kwargs,
               ) -> dict:

        tokenized_text = self.tokenizer(text, return_tensors='pt')
        tokenized_text = tokenized_text.to(self.device)

        seq_len = kwargs.get("seq_len", 10)
        tokens_len = len(tokenized_text)

        rng = mersenne_rng(hash_key)
        xi = np.array([rng.rand() for _ in range(seq_len * self.vocab_size)], dtype=np.float32).reshape(seq_len, self.vocab_size)
        test_result = self.distance_check(tokenized_text, tokens_len, xi)

        p_val = 0
        for run in range(n_runs):
            xi_alternative = np.random.rand(seq_len, self.vocab_size).astype(np.float32)
            null_result = self.distance_check(tokenized_text, tokens_len, xi_alternative)

            # assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result

        return {"z_score": (p_val + 1.0) / (n_runs + 1.0)}

    def distance_check(self,
                       tokens,
                       prompt_len: int,
                       xi: list,
                       gamma: float = 0.0):
        m = len(tokens)
        n = len(xi)
        tokens = tokens.input_ids.cpu().detach().numpy().squeeze()

        A = np.empty((m - (prompt_len - 1), n))
        for i in range(m - (prompt_len - 1)):
            for j in range(n):
                A[i][j] = levenshtein(tokens[i:i + prompt_len], xi[(j + np.arange(prompt_len)) % n], gamma)

        return np.min(A)
