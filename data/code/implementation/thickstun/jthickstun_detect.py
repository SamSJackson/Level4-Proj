import numpy as np

from data.code.static.mersenne import mersenne_rng
from data.code.static.levenshtein import levenshtein


def permutation_test(tokens, key: int, n: int, k: int, vocab_size: int, n_runs=100):
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


def detect(tokens, k: int, xi: list, gamma: float = 0.0):
    m = len(tokens)
    n = len(xi)

    A = np.empty((m - (k - 1), n))
    for i in range(m - (k - 1)):
        for j in range(n):
            A[i][j] = levenshtein(tokens[i:i + k], xi[(j + np.arange(k)) % n], gamma)

    return np.min(A)
