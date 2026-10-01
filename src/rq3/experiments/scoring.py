"""Score tokenised RQ3 codas in explicit base-2 units."""


def score_coda_bits(model, token_sequence):
    """Return total coda log probability and its per-predicted-symbol value."""
    coda_log2_probability = model.log2_probability(token_sequence)
    token_count = len(token_sequence) - 1  # BOS is context only; EOS is predicted.
    if token_count <= 0:
        raise ValueError("A scored coda must contain at least one predicted symbol")
    return {
        "token_count": token_count,
        "coda_log2_probability": coda_log2_probability,
        "bits_per_coda": coda_log2_probability,
        "bits_per_symbol": coda_log2_probability / token_count,
    }
