import os, sys, argparse, time

import numpy as np
from transformers import AutoTokenizer
from data.code.implementation.newthickstun.mersenne import mersenne_rng

import pyximport

pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs': np.get_include()})
from data.code.implementation.newthickstun.levenshtein import levenshtein


def permutation_test(tokens, key, n, k, vocab_size, n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n * vocab_size)], dtype=np.float32).reshape(n, vocab_size)
    test_result = detect(tokens, n, k, xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens, n, k, xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val + 1.0) / (n_runs + 1.0)


def detect(tokens, n, k, xi, gamma=0.0):
    m = len(tokens)
    n = len(xi)

    A = np.empty((m - (k - 1), n))
    for i in range(m - (k - 1)):
        for j in range(n):
            A[i][j] = levenshtein(tokens[i:i + k], xi[(j + np.arange(k)) % n], gamma)
    return np.min(A)


def main(args):
    with open(args.document, 'r') as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]

    t0 = time.time()
    pval = permutation_test(tokens, args.key, args.n, len(tokens), len(tokenizer))
    print('p-value: ', pval)
    print(f'(elapsed time: {time.time() - t0}s)')
