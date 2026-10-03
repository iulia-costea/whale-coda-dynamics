"""Fixed, data-independent tokenisation for RQ3 within-coda ICI sequences."""
import math


TOKENIZER_ID = "fixed-50ms-v1"
BIN_WIDTH_S = 0.05
MIN_ICI_S = 0.005
MAX_ICI_S = 1.0
ICI_SYMBOLS = tuple(range(20))
BOS = "BOS"
EOS = "EOS"


def ici_to_symbol(ici_s):
    """Map one in-range ICI to its fixed 50 ms symbol (0 through 19)."""
    if type(ici_s) not in (int, float) or not math.isfinite(ici_s):
        raise ValueError(f"ICI must be finite: {ici_s!r}")
    if not MIN_ICI_S <= ici_s < MAX_ICI_S:
        raise ValueError(f"ICI must be in [{MIN_ICI_S}, {MAX_ICI_S}): {ici_s!r}")
    return int(math.floor(ici_s / BIN_WIDTH_S))


def tokenize_ici_sequence(ici_sequence_s):
    """Wrap the fixed symbols for one coda in explicit BOS and EOS tokens."""
    return (BOS, *(ici_to_symbol(ici_s) for ici_s in ici_sequence_s), EOS)
