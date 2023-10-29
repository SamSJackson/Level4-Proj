class BaseGenerator:

    def __init__(
            self,
            vocab,
            model,
            tokenizer,
            hash_key,
            **kwargs,
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hash_key = hash_key
        self.rng = None
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device

        self.gamma = kwargs.get('gamma')
        self.delta = kwargs.get('delta')

    def generate(self,
                 prompt,
                 *args,
                 **kwargs) -> str:
        return f"{prompt}: ABSTRACT CLASS"

