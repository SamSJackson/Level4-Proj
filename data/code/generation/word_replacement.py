import random, re, tqdm
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
from flair.data import Sentence, Token
from flair.models import SequenceTagger
from nltk.corpus import wordnet as wnet

def word_replace_and_get_path(
        replacement: int,
        input_dir: str,
        no_paraphrases: int = 3,
        target_dir: str = "../../processed/"
):
    date = datetime.now().strftime("%d_%m_%Y")

    seq_tagger = SequenceTagger.load('flair/pos-english')
    df = pd.read_csv(input_dir)

    kgw_watermarked = df["kgw-watermarked"]
    non_watermarked = df["non-watermarked"]

    def parse_lemma(lemma_word: str) -> str:
        return lemma_word.replace("_", " ")

    def get_wordnet_pos(treebank_tag):

        if treebank_tag.startswith('J'):
            return wnet.ADJ
        elif treebank_tag.startswith('V'):
            return wnet.VERB
        elif treebank_tag.startswith('N'):
            return wnet.NOUN
        elif treebank_tag.startswith('R'):
            return wnet.ADV
        else:
            return ''

    def replace_single_word(fl_token: Token) -> str:
        text = fl_token.text
        if len(fl_token.labels) == 0:
            return text

        syns = wnet.synsets(text, pos=get_wordnet_pos(fl_token.labels[0].value))
        if len(syns) == 0:
            return text

        sy_word = random.choice(syns)
        lemma_word = random.choice(sy_word.lemma_names())
        return parse_lemma(lemma_word)

    def replace_words(text: str, percent: int=replacement) -> str:
        fl_sentence = Sentence(text)
        to_mask = int(np.ceil(len(fl_sentence.tokens) * (percent / 100)))

        mask = np.ones(len(fl_sentence.tokens))
        mask[:to_mask] = 0
        np.random.shuffle(mask)

        seq_tagger.predict(fl_sentence)
        altered_sentence = ""

        for token, masked in zip(fl_sentence.tokens, mask):
            word = replace_single_word(token) if masked else token.text
            altered_sentence += f"{word} "

        altered_sentence = re.sub(r' (?=\W)', '', altered_sentence)
        return altered_sentence


    for i in range(no_paraphrases):
        kgw_replaced = []
        nwmarked_replaced = []

        for text in tqdm.tqdm(kgw_watermarked):
            response = replace_words(text)
            kgw_replaced.append(response)

        for text in tqdm.tqdm(non_watermarked):
            response = replace_words(text)
            nwmarked_replaced.append(response)

        df[f"pp-kgw-word-{i+1}"] = kgw_replaced
        df[f"pp-unwatermarked-word-{i+1}"] = nwmarked_replaced

        kgw_watermarked = kgw_replaced.copy()
        non_watermarked = nwmarked_replaced.copy()

    output_path = Path(f"{target_dir}/paraphrased/", parents=True, exist_ok=True)
    output_file = f"word_replaced_{df.shape[0]}_{date}.csv"
    df.to_csv(output_path / output_file, index=False)

    del seq_tagger

    return output_path / output_file

if __name__ == "__main__":
    word_replace_and_get_path(
        replacement=20
    )