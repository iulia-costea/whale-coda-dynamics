"""Small autoregressive sequence models for primary RQ3 calibration."""
import math
from collections import Counter, defaultdict

from rq3.preprocessing.tokenize import EOS


PREDICTION_VOCABULARY = tuple(range(20)) + (EOS,)


class SmoothedNGram:
    """Fixed-order additive-smoothed n-gram over ICI symbols and EOS."""

    def __init__(self, context_order, alpha):
        self.context_order = context_order
        self.alpha = alpha
        self.counts = defaultdict(Counter)

    def fit(self, token_sequences):
        """Fit conditional token counts from full BOS-to-EOS sequences."""
        self.counts = defaultdict(Counter)
        sequence_count = 0
        for sequence in token_sequences:
            if len(sequence) < 3 or sequence[-1] != EOS:
                raise ValueError("N-gram training expects a non-empty BOS-to-EOS sequence")
            sequence_count += 1
            for index, token in enumerate(sequence[1:], start=1):
                context = tuple(sequence[max(0, index - self.context_order):index]) if self.context_order else ()
                self.counts[context][token] += 1
        if not sequence_count:
            raise ValueError("Cannot fit an n-gram with no coda sequences")
        return self

    def log2_probability(self, token_sequence):
        """Return total base-2 log probability for one full BOS-to-EOS sequence."""
        if len(token_sequence) < 3 or token_sequence[-1] != EOS:
            raise ValueError("N-gram scoring expects a non-empty BOS-to-EOS sequence")
        total = 0.0
        vocabulary_size = len(PREDICTION_VOCABULARY)
        for index, token in enumerate(token_sequence[1:], start=1):
            context = tuple(token_sequence[max(0, index - self.context_order):index]) if self.context_order else ()
            counts = self.counts[context]
            probability = (counts[token] + self.alpha) / (sum(counts.values()) + self.alpha * vocabulary_size)
            total += math.log2(probability)
        return total
