"""Inputs, encoding, held-out plans and leakage checks for the pooled RQ3 LSTM ([R3-8]).

Codas, folds and tokens come from Vlad's RQ3 code, unchanged: ``read_csv``, ``parse_codas`` and
``build_fold_lookup`` from ``rq3.experiments.data`` and ``tokenize_ici_sequence`` from
``rq3.preprocessing.tokenize``. A tagged model replaces the tokeniser's ``BOS`` with
``BOS_<clan>``; nothing else about a coda's symbols changes.

Every model, in every arm, has the same input vocabulary: the 20 interval symbols, the shared
``BOS``, one ``BOS_<clan>`` per clan and a padding symbol. Rows a model never sees get no gradient,
so models that differ only in their start symbol start from identical weights.
"""
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from whale_coda_dynamics.preprocessing.io_utils import read_json
from whale_coda_dynamics.rq3.experiments.data import (
    CORE_COLUMNS,
    FOLD_COLUMNS,
    SELF_SCORE_COLUMNS,
    build_fold_lookup,
    parse_codas,
    read_csv,
)
from whale_coda_dynamics.rq3.preprocessing.tokenize import BOS, EOS, ICI_SYMBOLS, TOKENIZER_ID, tokenize_ici_sequence

MAX_SYMBOLS = 10                     # 3-10 clicks: 2-9 intervals plus EOS
N_OUTPUTS = len(ICI_SYMBOLS) + 1     # every interval symbol and EOS
EOS_OUTPUT = len(ICI_SYMBOLS)
BOS_INPUT = len(ICI_SYMBOLS)

REQUIRED_KEYS = {
    'codas_input', 'folds_input', 'ngram_self_scores_input', 'ngram_calibration_input',
    'ngram_matrix_scores_input', 'ngram_matrix_summary_input', 'output_directory', 'checkpoint_directory',
    'report_output', 'tokenizer_id', 'clans', 'model', 'gate_rerun', 'pooled_alphas', 'arm5',
    'capacity_multiplier', 'bootstrap_draws', 'bootstrap_seed', 'device', 'workers', 'pilot',
    'require_clean_tree', 'smoke_test',
}
MODEL_KEYS = {'embedding_dim', 'batch_size', 'learning_rate', 'max_epochs', 'patience', 'hidden_grid',
              'dropout_grid', 'selection_seed', 'refit_seeds', 'eval_batch_size'}


def load_config(config_path):
    """Read and strictly check the run configuration; paths resolve against its parent.parent."""
    path = Path(config_path).resolve()
    config = read_json(path)
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f'rq3_neural config is missing keys: {sorted(missing)}')
    if config['tokenizer_id'] != TOKENIZER_ID:
        raise ValueError(f'[R3-8] uses the [R3-1] tokeniser {TOKENIZER_ID}, not {config["tokenizer_id"]!r}')
    model = config['model']
    if set(model) != MODEL_KEYS:
        raise ValueError(f'rq3_neural model settings must be exactly {sorted(MODEL_KEYS)}')
    if not model['hidden_grid'] or not model['dropout_grid'] or not model['refit_seeds']:
        raise ValueError('The grid and the refit seeds must be non-empty')
    if set(config['gate_rerun']) != {'hidden_grid', 'learning_rates', 'patience'}:
        raise ValueError('gate_rerun must give hidden_grid, learning_rates and patience')
    arm5 = config['arm5']
    if set(arm5) != {'shares', 'draws', 'learning_rates', 'reduced_shares', 'reduced_draws'}:
        raise ValueError('arm5 must give shares, draws, learning_rates, reduced_shares and reduced_draws')
    if any(not 0 < f <= 1 for f in arm5['shares'] + arm5['reduced_shares']):
        raise ValueError('arm5 shares must lie in (0, 1]')
    if arm5['draws'] > len(model['refit_seeds']) or arm5['reduced_draws'] > arm5['draws']:
        raise ValueError("arm5 draw d fine-tunes arm 4's refit with seed d: draws must not exceed the refit seeds")
    if sorted(config['clans']) != config['clans'] or len(set(config['clans'])) != len(config['clans']):
        raise ValueError('clans must be listed once each, sorted')
    if config['device'] not in {'auto', 'cpu', 'cuda'}:
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    return path, path.parent.parent, config


def model_tokens(tokens, clan, tagged):
    """The [R3-1] tokens with BOS replaced by BOS_<clan> when tagged; nothing else changes."""
    if not tokens or tokens[0] != BOS or tokens[-1] != EOS:
        raise ValueError(f'Expected a BOS ... EOS token sequence, got {tokens!r}')
    return (f'BOS_{clan}' if tagged else BOS, *tokens[1:])


def input_vocabulary(clans):
    vocab = {symbol: symbol for symbol in ICI_SYMBOLS}
    vocab[BOS] = BOS_INPUT
    for offset, clan in enumerate(clans):
        vocab[f'BOS_{clan}'] = BOS_INPUT + 1 + offset
    pad = BOS_INPUT + 1 + len(clans)
    return vocab, pad


def encode(codas, clans):
    """Padded input/target arrays for every coda, untagged and tagged.

    Inputs are the start symbol and the intervals; targets are the intervals and EOS, so a coda
    with n intervals has n + 1 predicted symbols: Vlad's ``token_count`` (len(tokens) - 1)."""
    vocab, pad = input_vocabulary(clans)
    clan_code = {clan: i for i, clan in enumerate(clans)}
    n = len(codas)
    x = {False: np.full((n, MAX_SYMBOLS), pad, dtype=np.int64), True: np.full((n, MAX_SYMBOLS), pad, dtype=np.int64)}
    y = np.zeros((n, MAX_SYMBOLS), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    clan_index = np.zeros(n, dtype=np.int64)
    for i, coda in enumerate(codas):
        tokens = coda['tokens']
        targets = [EOS_OUTPUT if t == EOS else t for t in tokens[1:]]
        if any(t not in range(N_OUTPUTS) for t in targets):
            raise ValueError(f"Unexpected symbol in {coda['rq3_coda_id']}: {tokens!r}")
        if not 1 <= len(targets) <= MAX_SYMBOLS:
            raise ValueError(f"{coda['rq3_coda_id']} has {len(targets)} predicted symbols")
        for tagged in (False, True):
            inputs = [vocab[t] for t in model_tokens(tokens, coda['clan'], tagged)[:-1]]
            x[tagged][i, :len(inputs)] = inputs
        y[i, :len(targets)] = targets
        lengths[i] = len(targets)
        clan_index[i] = clan_code[coda['clan']]
    encoded = dict(x_untagged=x[False], x_tagged=x[True], y=y, lengths=lengths, clan_index=clan_index,
                   pad=pad, n_inputs=pad + 1, clans=list(clans))
    check_tag_inputs(encoded)
    return encoded


def check_tag_inputs(encoded):
    """Tagged and untagged inputs differ only in the start symbol; targets are shared."""
    xu, xt = encoded['x_untagged'], encoded['x_tagged']
    if xu.shape != xt.shape or not np.array_equal(xu[:, 1:], xt[:, 1:]):
        raise ValueError('Tagged and untagged inputs differ beyond the start symbol')
    if not np.all(xu[:, 0] == BOS_INPUT):
        raise ValueError('Untagged inputs must start with the shared BOS')
    if not np.array_equal(xt[:, 0], BOS_INPUT + 1 + encoded['clan_index']):
        raise ValueError("Tagged inputs must start with the coda's own BOS_<clan>")


def array_sha256(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Context:
    """Everything a stage needs about the data: codas, folds, encodings and index sets."""

    def __init__(self, config_path):
        self.config_path, self.root, self.config = load_config(config_path)
        config = self.config
        self.codas = parse_codas(read_csv(self.root / config['codas_input'], CORE_COLUMNS))
        self.fold_rows = read_csv(self.root / config['folds_input'], FOLD_COLUMNS)
        fold_of = build_fold_lookup(self.codas, self.fold_rows)
        for coda in self.codas:
            coda['fold'] = fold_of[coda['rq3_coda_id']]
            coda['tokens'] = tokenize_ici_sequence(coda['ici_sequence_s'])
        clans = config['clans']
        if sorted({coda['clan'] for coda in self.codas}) != clans:
            raise ValueError(f'Configured clans {clans} differ from the clans in the codas file')
        self.clans = clans
        self.encoded = encode(self.codas, clans)
        self.ids = [coda['rq3_coda_id'] for coda in self.codas]
        self.group = [coda['group_id'] for coda in self.codas]
        self.clan_of = [coda['clan'] for coda in self.codas]
        self.fold_of = np.array([coda['fold'] for coda in self.codas], dtype=np.int64)
        self.n_symbols = self.encoded['lengths']
        self.n_folds = {clan: len({c['fold'] for c in self.codas if c['clan'] == clan}) for clan in clans}
        self.index = {}
        for clan in clans:
            for fold in range(self.n_folds[clan]):
                self.index[clan, fold] = np.array(
                    [i for i, c in enumerate(self.codas) if c['clan'] == clan and c['fold'] == fold], dtype=np.int64)
        self.vlad_token_count = load_vlad_token_counts(self.root / config['ngram_self_scores_input'], self.ids)
        check_n_symbols('encoding', np.arange(len(self.ids)), self.n_symbols, self)

    def clan_indices(self, clan, folds=None):
        folds = range(self.n_folds[clan]) if folds is None else folds
        parts = [self.index[clan, fold] for fold in folds]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def path(self, key):
        return self.root / self.config[key]


def load_vlad_token_counts(path, ids):
    """Vlad's per-coda token_count from the own-clan n-gram scores; one row for every coda."""
    rows = read_csv(path, SELF_SCORE_COLUMNS)
    counts = {row['rq3_coda_id']: int(row['token_count']) for row in rows}
    if len(counts) != len(rows) or set(counts) != set(ids):
        raise ValueError("The n-gram self scores must have exactly one row for every coda")
    return counts


# ---------------------------------------------------------------- gate 0

def gate0_rows(ctx):
    """Per-clan and per-fold coda counts, from the codas file and from rq3_folds.csv."""
    from_codas = Counter((coda['clan'], coda['fold']) for coda in ctx.codas)
    from_file = Counter()
    for row in ctx.fold_rows:
        from_file[row['clan'], int(row['fold'])] += int(row['n_codas'])
    rows = []
    for clan in ctx.clans:
        for fold in range(ctx.n_folds[clan]):
            rows.append(dict(clan=clan, fold=fold, codas_file=from_codas[clan, fold], fold_file=from_file[clan, fold],
                             encoded=len(ctx.index[clan, fold])))
        rows.append(dict(clan=clan, fold='all', codas_file=sum(from_codas[clan, f] for f in range(ctx.n_folds[clan])),
                         fold_file=sum(v for (c, _), v in from_file.items() if c == clan),
                         encoded=len(ctx.clan_indices(clan))))
    if set(from_codas) != set(from_file):
        raise ValueError('Gate 0: the codas file and rq3_folds.csv cover different clan/fold cells')
    for row in rows:
        row['match'] = row['codas_file'] == row['fold_file'] == row['encoded']
        if not row['match']:
            raise ValueError(f"Gate 0 failed for {row['clan']} fold {row['fold']}: {row}")
    return rows


# ---------------------------------------------------------------- held-out plans

def own_clan_plan(ctx, clan, fold):
    """Arm 1 (and arm 5): clan c, fold k held out, inner validation on (k + 1) mod K_c."""
    k_c = ctx.n_folds[clan]
    inner = (fold + 1) % k_c
    rest = [f for f in range(k_c) if f not in (fold, inner)]
    return dict(clans=(clan,), held_folds={clan: fold}, inner_folds={clan: inner},
                held=ctx.index[clan, fold], val=ctx.index[clan, inner],
                select_train=ctx.clan_indices(clan, rest),
                refit_train=ctx.clan_indices(clan, [f for f in range(k_c) if f != fold]),
                refit_folds={clan: [f for f in range(k_c) if f != fold]})


def pooled_plan(ctx, run):
    """Arms 2 and 3, run k: fold k of every clan that has one is held out; inner validation on
    fold (k + 1) mod K_c of every clan; the refit trains on every fold except the held-out one."""
    held, val, select, refit, held_folds, inner_folds, refit_folds = [], [], [], [], {}, {}, {}
    for clan in ctx.clans:
        k_c = ctx.n_folds[clan]
        held_fold = run if run < k_c else None
        inner = (run + 1) % k_c
        held_folds[clan], inner_folds[clan] = held_fold, inner
        if held_fold is not None:
            held.append(ctx.index[clan, held_fold])
        val.append(ctx.index[clan, inner])
        select.append(ctx.clan_indices(clan, [f for f in range(k_c) if f not in (held_fold, inner)]))
        refit_folds[clan] = [f for f in range(k_c) if f != held_fold]
        refit.append(ctx.clan_indices(clan, refit_folds[clan]))
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)  # noqa: E731
    return dict(clans=tuple(ctx.clans), held_folds=held_folds, inner_folds=inner_folds, held=cat(held),
                val=cat(val), select_train=cat(select), refit_train=cat(refit), refit_folds=refit_folds)


def pooled_runs(ctx):
    return list(range(max(ctx.n_folds.values())))


def loo_plan(ctx, left_out):
    """Arm 4: the other eight clans, inner validation on fold 0 of each, refit on all their codas;
    scores every coda of the left-out clan."""
    clans = tuple(c for c in ctx.clans if c != left_out)
    val = np.concatenate([ctx.index[c, 0] for c in clans])
    select = np.concatenate([ctx.clan_indices(c, range(1, ctx.n_folds[c])) for c in clans])
    refit = np.concatenate([ctx.clan_indices(c) for c in clans])
    plan = dict(clans=clans, left_out=left_out, held_folds={left_out: 'all'}, inner_folds={c: 0 for c in clans},
                held=ctx.clan_indices(left_out), val=val, select_train=select, refit_train=refit,
                refit_folds={c: list(range(ctx.n_folds[c])) for c in clans})
    check_excludes_clan(f'arm4 without {left_out}', ctx, left_out, plan['val'], plan['select_train'], plan['refit_train'])
    return plan


# ---------------------------------------------------------------- checks (raise, never work around)

def check_disjoint(label, ctx, scored, *seen):
    """A model's training and validation codas are disjoint from the codas it scores."""
    seen = [np.asarray(s, dtype=np.int64) for s in seen if s is not None]
    seen = np.concatenate(seen) if seen else np.zeros(0, dtype=np.int64)
    overlap = np.intersect1d(seen, np.asarray(scored, dtype=np.int64))
    if overlap.size:
        examples = [ctx.ids[i] for i in overlap[:3]]
        raise ValueError(f'{label}: {overlap.size} scored codas were also used in training or validation, e.g. {examples}')


def check_train_val_disjoint(label, ctx, train, val):
    overlap = np.intersect1d(np.asarray(train), np.asarray(val))
    if overlap.size:
        raise ValueError(f'{label}: {overlap.size} codas are in both training and inner validation, '
                         f'e.g. {[ctx.ids[i] for i in overlap[:3]]}')


def check_excludes_clan(label, ctx, clan, *index_sets):
    """A leave-one-out model contains no coda of the left-out clan."""
    code = ctx.clans.index(clan)
    for index in index_sets:
        if index is not None and np.any(ctx.encoded['clan_index'][np.asarray(index, dtype=np.int64)] == code):
            raise ValueError(f'{label}: contains codas of the left-out clan {clan}')


def check_n_symbols(label, index, n_symbols, ctx):
    """Every scored coda's n_symbols equals the encoding's and Vlad's token_count."""
    index = np.asarray(index, dtype=np.int64)
    n_symbols = np.asarray(n_symbols, dtype=np.int64)
    if index.shape != n_symbols.shape:
        raise ValueError(f'{label}: {index.size} codas but {n_symbols.size} symbol counts')
    expected = ctx.n_symbols[index]
    vlad = np.array([ctx.vlad_token_count[ctx.ids[i]] for i in index], dtype=np.int64)
    bad = np.flatnonzero((n_symbols != expected) | (n_symbols != vlad))
    if bad.size:
        i = index[bad[0]]
        raise ValueError(f'{label}: n_symbols for {ctx.ids[i]} is {n_symbols[bad[0]]}, encoding {ctx.n_symbols[i]}, '
                         f'Vlad token_count {ctx.vlad_token_count[ctx.ids[i]]}')
