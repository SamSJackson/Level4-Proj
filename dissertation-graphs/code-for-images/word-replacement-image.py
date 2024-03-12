import random, re
import numpy as np

from flair.data import Sentence, Token
from flair.models import SequenceTagger
from nltk.corpus import wordnet as wnet

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

def replace_words(model, text: str, percent: int=20) -> str:
    fl_sentence = Sentence(text)
    to_mask = int(np.ceil(len(fl_sentence.tokens) * (percent / 100)))

    mask = np.ones(len(fl_sentence.tokens))
    mask[:to_mask] = 0
    np.random.shuffle(mask)

    model.predict(fl_sentence)
    altered_sentence = ""

    for token, masked in zip(fl_sentence.tokens, mask):
        word = replace_single_word(token) if masked else token.text
        altered_sentence += f"{word} "

    altered_sentence = re.sub(r' (?=\W)', '', altered_sentence)
    return altered_sentence

if __name__ == "__main__":
    seq_tagger = SequenceTagger.load('flair/pos-english')

    text = "He is confiding in me!"
    replaced_text = replace_words(seq_tagger, text)

    print(f"Original Text: {text}")
    print(f"Word-Replaced: {replaced_text}")

