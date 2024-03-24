import numpy as np

from src.code.implementation.wieting.utils import Example

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class Similarity(object):

    def __init__(self, batch_size: int, entok, sp, model, lower_case, tokenize):
        self.batch_size = 32
        self.entok = entok
        self.sp = sp
        self.model = model
        self.lower_case = lower_case
        self.tokenize = tokenize
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))


    def batcher(self, batch):
        new_batch = []
        for p in batch:
            if self.tokenize:
                tok = self.entok.tokenize(p, escape=False)
                p = " ".join(tok)
            if self.lower_case:
                p = p.lower()
            p = self.sp.EncodeAsPieces(p)
            p = " ".join(p)
            p = Example(p, self.lower_case)
            p.populate_embeddings(self.model.vocab, self.model.zero_unk, self.model.ngrams)
            new_batch.append(p)
        x, l = self.model.torchify_batch(new_batch)
        vecs = self.model.encode(x, l)
        return vecs.detach().cpu().numpy()

    def score(self, input_text):
        input1 = []
        input2 = []
        for text1, text2 in input_text:
            input1.append(text1.strip())
            input2.append(text2.strip())

        sys_scores = []
        for ii in range(0, len(input1), self.batch_size):
            batch1 = input1[ii:ii + self.batch_size]
            batch2 = input2[ii:ii + self.batch_size]

            # we assume get_batch already throws out the faulty ones
            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = self.batcher(batch1)
                enc2 = self.batcher(batch2)

                for kk in range(enc2.shape[0]):
                    sys_score = self.similarity(enc1[kk], enc2[kk])
                    sys_scores.append(sys_score)

        return sys_scores

def make_sim_object(batch_size, entok, model):
    sim = Similarity(batch_size=batch_size,entok=entok,
                     sp=model.sp,model=model,
                     lower_case=model.args.lower_case,
                     tokenize=model.args.tokenize)
    return sim

def evaluate(sim_object: Similarity, source_document: str, paraphrases: list[str]) -> np.array:
    '''
        Comparing each one individual.
        Would prefer comparing each one against source as one function.
        Try and improve later.
    '''
    pairs = [[(source_document, paraphrase)] for paraphrase in paraphrases]
    return [sim_object.score(pair)[0] for pair in pairs]



