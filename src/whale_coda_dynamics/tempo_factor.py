"""Partner-tempo ablation (DEC-24, PLAN [R2-8]): the chain rule's tempo factor as its own model.

The target is a coda's tempo class (the KDE classes, DEC-22), given the coda's own pattern, which is always
present: its Sharma rhythm in the primary run, its guy_coda_type in the secondary. Blocks:
- own (in every condition but the floor): the whale's last 3 tempo classes and last 3 patterns, one-hot by
  lag, plus a shared copy weight beta on 1[tempo = own last tempo];
- T, partner tempo: the partner's last 3 tempo classes, plus a shared copy weight gamma on
  1[tempo = partner's last tempo];
- P, partner pattern: the partner's last 3 patterns;
- S: RQ1's interactional state, encoded by logistic.Encoder exactly as in logistic v4.
Conditions: own with every subset of {T, P, S} (eight, named own, own_T, ..., own_T_P_S), and the floor, the
coda's own pattern alone. DEC-24's base is own_S.

Labels: every label in the target files is '<pattern>|<tempo>' (inventories D and E, PLAN [R2-1], [R2-2]).
Tempo classes have no UNK. Patterns follow RQ1's UNK policy per fold (fit_vocabulary on the training days'
codas, PLAN [M6]): one vocabulary per factor, shared by the coda's own pattern and every lag of both whales.
Histories are the codas completed before the target's onset ([D5]): A_history_ids and B_history_ids.

Fitting: logistic v4's procedure (logistic.grid_search_l2): every non-S column, both copy weights and the
intercept share one penalty, S has its own, both chosen on the validation day from the same grid; the floor
has one penalty. Selection (DEC-23's v5): per fold and condition, the candidates are the fits of every
sub-condition (T, P and S each on or absent; own always present). The highest validation-day log-likelihood
wins (logistic_v5.select_max; exact ties: fewer blocks, then the order T, P, S), with DEC-21's one-SE rule as
the sensitivity. The floor's only candidate is its own fit. The own fits (no absent candidates) are reported
as 'own_fit'.

Contrasts, in bits per coda of the tempo class, paired: partner tempo alone (own_T_S - own_S), partner
pattern alone (own_P_S - own_S), tempo beyond pattern (own_T_P_S - own_P_S), pattern beyond tempo
(own_T_P_S - own_T_S); day-bootstrap CI and the exact two-sided sign-flip p. Stop rules (DEC-23) on the
family's 13 add-one-block contrasts (the 12 pairs among the eight conditions, and own - floor), for the own
fits, v5 and v5 one-SE; DEC-23's rule on v5 governs. A descriptive breakdown by turn category, with no tests.

Gates, all before any test-day number is computed; nothing is written if one fails:
1. labels: every pattern equals the coarse inventory's label (C's rhythm, RQ1's type) and every tempo is a
   configured class, equal across runs; every other field the model reads equals the coarse file's;
2. folds: every run's folds (days, roles, comparison IDs) equal the coarse inventory's and RQ1's, and each
   fold's pattern vocabulary equals the coarse inventory's fold vocabulary;
3. validation monotonicity: under the max rule every add-one-block contrast among the eight conditions is
   >= 0 on every validation day (true by construction; checked anyway).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import scipy

from .evaluation import classification_metrics, day_bootstrap_bits, log_likelihood_bits, sign_flip_p
from .logistic import Encoder, format_ci, grid_search_l2, probabilities
from .logistic_v5 import add_input_pairs, day_sums, dec23_stop_rules, rebuild_predictions, select_max, select_one_se, subconditions
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text
from .preprocessing.vocabulary import fit_vocabulary

BLOCKS = ('T', 'P', 'S')
FLOOR = 'floor'
# Factor streams, each with the factor whose vocabulary it uses.
STREAMS = (('own_tempo', 'tempo'), ('own_pattern', 'pattern'), ('partner_tempo', 'tempo'), ('partner_pattern', 'pattern'))
TURN_CATEGORIES = ('overlap', 'switch', 'self_continuation')
RULES = ('max', 'one_se')
MODELS = (('own_fit', 'own fit (no absent candidates)'), ('v5', 'v5 (max)'), ('v5_one_se', 'v5 (one-SE)'))
GOVERNING_MODEL = 'v5'
FOLD_KEYS = ('fold_id', 'test_day', 'validation_day', 'day_splits', 'comparison_ids')


def condition_name(blocks):
    """'own' followed by the blocks in BLOCKS order: () -> 'own', ('S', 'T') -> 'own_T_S'."""
    if any(b not in BLOCKS for b in blocks) or len(set(blocks)) != len(blocks):
        raise ValueError(f'Unknown or repeated blocks {blocks}')
    return '_'.join(('own',) + tuple(sorted(blocks, key=BLOCKS.index)))


def condition_blocks(condition):
    """The blocks of a condition (own is implicit), or None for the floor."""
    if condition == FLOOR:
        return None
    parts = condition.split('_')
    if parts[0] != 'own' or condition_name(parts[1:]) != condition:
        raise ValueError(f'Unknown condition {condition!r}')
    return tuple(parts[1:])


LATTICE = [condition_name(s) for s in subconditions(BLOCKS, BLOCKS)]
CONDITIONS = [FLOOR] + LATTICE
CONTRASTS = {'partner_tempo': ('own_T_S', 'own_S'), 'partner_pattern': ('own_P_S', 'own_S'),
             'tempo_beyond_pattern': ('own_T_P_S', 'own_P_S'), 'pattern_beyond_tempo': ('own_T_P_S', 'own_T_S')}


def stop_family():
    """The 13 add-one-block pairs (bigger, smaller): the 12 among the eight conditions, then own - floor."""
    return add_input_pairs(LATTICE) + [('own', FLOOR)]


def check_name(big, small):
    return f'add_block:{big}-{small}'


# ---------------------------------------------------------------- labels and folds

def split_label(label):
    """'<pattern>|<tempo>' -> (pattern, tempo)."""
    parts = label.split('|') if isinstance(label, str) else []
    if len(parts) != 2 or not all(parts):
        raise ValueError(f'Expected a <pattern>|<tempo> label, got {label!r}')
    return parts[0], parts[1]


def factor_labels(rows, coarse_rows, tempo_classes, s_fields):
    """Gate 1 for one run: (pattern, tempo) per coda ID from the fine inventory's raw labels. The pattern
    must equal the coarse inventory's raw label, the tempo a configured class, and every field the model
    reads the coarse file's."""
    if [r['coda_id'] for r in rows] != [r['coda_id'] for r in coarse_rows]:
        raise ValueError('The fine and coarse target files hold different codas')
    keys = ('recording_date', 'comparison_eligible', 'A_history_ids', 'B_history_ids', 'turn_category', *s_fields)
    labels = {}
    for row, coarse in zip(rows, coarse_rows):
        pattern, tempo = split_label(row['Ct_raw'])
        if pattern != coarse['Ct_raw']:
            raise ValueError(f"{row['coda_id']}: pattern {pattern!r} differs from the coarse label {coarse['Ct_raw']!r}")
        if tempo not in tempo_classes:
            raise ValueError(f"{row['coda_id']}: tempo {tempo!r} is not a configured class")
        if any(row[k] != coarse[k] for k in keys):
            raise ValueError(f"{row['coda_id']}: fields differ between the fine and coarse target files")
        labels[row['coda_id']] = (pattern, tempo)
    return labels


def check_folds(manifest, *references):
    """Gate 2, first part: the folds and the S fields equal every reference manifest's."""
    folds = [{k: f[k] for k in FOLD_KEYS} for f in manifest['folds']]
    for reference in references:
        if folds != [{k: f[k] for k in FOLD_KEYS} for f in reference['folds']]:
            raise ValueError('Folds differ between manifests')
        if (manifest['input_conditions']['S'] != reference['input_conditions']['S']
                or manifest['preparation_configuration'] != reference['preparation_configuration']):
            raise ValueError('S fields or preparation settings differ between manifests')


def prepare_factor_fold(rows, labels, fold, coarse_fold, preparation, s_fields):
    """One fold's comparison rows per split with their factor streams, and its pattern vocabulary policy.
    The pattern vocabulary is RQ1's policy on the patterns of the training days' codas; it must equal the
    coarse inventory's fold vocabulary (gate 2, second part)."""
    mapped, policy = prepare_fold(rows, fold, preparation)
    if policy != fold['vocabulary_policy']:
        raise ValueError('Recomputed fold policy differs from manifest')
    pattern_policy = fit_vocabulary([dict(coda_id=r['coda_id'], split='train', coda_type=labels[r['coda_id']][0])
                                     for r in mapped if r['split'] == 'train'],
                                    preparation['unk_min_count'], preparation['unk_token'])
    if pattern_policy != {k: coarse_fold['vocabulary_policy'][k] for k in pattern_policy}:
        raise ValueError(f"{fold['fold_id']}: pattern vocabulary differs from the coarse inventory's")
    unk = pattern_policy['unk_token']
    known = set(pattern_policy['vocabulary']) - {unk}

    def pattern(coda_id):
        p = labels[coda_id][0]
        return p if p in known else unk

    prepared = dict(fold=fold, pattern_policy=pattern_policy)
    for split in ('train', 'validation', 'test'):
        out = [dict({f: r[f] for f in s_fields}, coda_id=r['coda_id'], split=split, recording_date=r['recording_date'],
                    turn_category=r['turn_category'], tempo=labels[r['coda_id']][1], pattern=pattern(r['coda_id']),
                    own_tempo=[labels[i][1] for i in r['A_history_ids']], own_pattern=[pattern(i) for i in r['A_history_ids']],
                    partner_tempo=[labels[i][1] for i in r['B_history_ids']],
                    partner_pattern=[pattern(i) for i in r['B_history_ids']])
               for r in mapped if r['split'] == split and r['comparison_eligible']]
        if sorted(r['coda_id'] for r in out) != fold['comparison_ids'][split]:
            raise ValueError(f"{fold['fold_id']}: {split} targets differ from the manifest")
        prepared[split] = out
    return prepared


# ---------------------------------------------------------------- encoder and fits

class FactorEncoder:
    """The coda's own pattern (one-hot), then each present stream's lags 1..order (one-hot in its factor's
    vocabulary; missing lags are zero vectors), then RQ1's S columns, then one intercept. S standardisation
    and turn categories are fitted on training rows only (logistic.Encoder). A label outside its vocabulary
    raises instead of becoming a zero vector."""

    def __init__(self, train, tempo_vocabulary, pattern_vocabulary, condition, s_fields, order=3):
        if not train or any(r['split'] != 'train' for r in train):
            raise ValueError('Encoder fitting requires training rows')
        if type(order) is not int or order < 1:
            raise ValueError('Invalid history order')
        for vocabulary in (tempo_vocabulary, pattern_vocabulary):
            if not vocabulary or len(set(vocabulary)) != len(vocabulary):
                raise ValueError('Vocabularies must be nonempty and without repeats')
        blocks = condition_blocks(condition)
        self.condition, self.order, self.s_fields = condition, order, list(s_fields)
        self.own, self.blocks = blocks is not None, blocks or ()
        self.vocabularies = {'tempo': list(tempo_vocabulary), 'pattern': list(pattern_vocabulary)}
        present = {'own_tempo': self.own, 'own_pattern': self.own, 'partner_tempo': 'T' in self.blocks,
                   'partner_pattern': 'P' in self.blocks}
        self.streams = [stream for stream, _ in STREAMS if present[stream]]
        self.s_encoder = Encoder(train, [], self.s_fields, order) if 'S' in self.blocks else None
        self.names = ['pattern:' + p for p in self.vocabularies['pattern']]
        for stream in self.streams:
            self.names += [f'{stream}:lag{lag}:{label}' for lag in range(1, order + 1) for label in self.vocabulary(stream)]
        s_names = self.s_encoder.names if self.s_encoder else []
        # DEC-20/DEC-24: S columns get the S penalty; everything else, the intercept included, the history penalty.
        self.timing_mask = np.array([False] * len(self.names) + [True] * len(s_names) + [False])
        self.names += s_names

    def vocabulary(self, stream):
        return self.vocabularies[dict(STREAMS)[stream]]

    def _known(self, label, factor):
        if label not in self.vocabularies[factor]:
            raise ValueError(f'{factor} label {label!r} is not in the fold vocabulary')

    def transform(self, rows):
        x = []
        for r in rows:
            self._known(r['pattern'], 'pattern')
            values = [float(r['pattern'] == p) for p in self.vocabularies['pattern']]
            for stream in self.streams:
                vocabulary = self.vocabulary(stream)
                for label in r[stream]:
                    self._known(label, dict(STREAMS)[stream])
                for lag in range(1, self.order + 1):
                    label = r[stream][-lag] if len(r[stream]) >= lag else None
                    values.extend(float(label == t) for t in vocabulary)
            x.append(values)
        x = np.asarray(x, dtype=float).reshape(len(rows), len(self.names) - (len(self.s_encoder.names) if self.s_encoder else 0))
        if self.s_encoder is not None:
            x = np.column_stack([x, self.s_encoder.transform(rows)[:, :-1]])
        return np.column_stack([x, np.ones(len(rows))])

    def copy_indices(self, rows):
        """Tempo-vocabulary index of the own last and the partner's last tempo class (-1 without one): beta's
        and gamma's class per row, or None where the copy weight is absent (own: the floor; gamma: no T)."""
        position = {t: i for i, t in enumerate(self.vocabularies['tempo'])}

        def last(stream):
            for r in rows:
                for label in r[stream]:
                    self._known(label, 'tempo')
            return np.array([position[r[stream][-1]] if r[stream] else -1 for r in rows], dtype=int)

        return (last('own_tempo') if self.own else None), (last('partner_tempo') if 'T' in self.blocks else None)

    def record(self):
        return dict(condition=self.condition, blocks=list(self.blocks), own=self.own, order=self.order,
                    tempo_vocabulary=self.vocabularies['tempo'], pattern_vocabulary=self.vocabularies['pattern'],
                    s_fields=self.s_fields if self.s_encoder else [],
                    categories=self.s_encoder.categories if self.s_encoder else [],
                    s_mean=self.s_encoder.mean.tolist() if self.s_encoder else [],
                    s_scale=self.s_encoder.scale.tolist() if self.s_encoder else [],
                    feature_names=self.names + ['intercept'])


def tempo_indices(rows, tempo_vocabulary):
    position = {t: i for i, t in enumerate(tempo_vocabulary)}
    return np.array([position[r['tempo']] for r in rows], dtype=int)


def fit_condition(prepared, condition, tempo_vocabulary, s_fields, config):
    """v4's procedure for one fold and condition, on training and validation days only. Returns the encoder,
    the chosen candidate, the whole grid and the chosen fit's per-target validation log-likelihoods."""
    encoder = FactorEncoder(prepared['train'], tempo_vocabulary, prepared['pattern_policy']['vocabulary'],
                            condition, s_fields, config['history_order'])
    train, validation = prepared['train'], prepared['validation']
    x_train, x_val = encoder.transform(train), encoder.transform(validation)
    if x_train.shape[1] != len(encoder.timing_mask):
        raise ValueError('Penalty mask and design matrix disagree')
    y_val = tempo_indices(validation, tempo_vocabulary)
    own_train, partner_train = encoder.copy_indices(train)
    own_val, partner_val = encoder.copy_indices(validation)
    chosen, grid = grid_search_l2(x_train, tempo_indices(train, tempo_vocabulary), x_val, y_val, len(tempo_vocabulary),
                                  encoder.timing_mask, config, own_train, partner_train, own_val, partner_val)
    p_val = probabilities(x_val, chosen['w'], chosen['beta'], chosen['gamma'], own_val, partner_val)
    bits = np.log2(p_val[np.arange(len(y_val)), y_val])
    if float(np.mean(bits)) != chosen['validation_log_likelihood_bits']:
        raise ValueError(f"{prepared['fold']['fold_id']}, {condition}: validation log-likelihood is not reproducible")
    return dict(condition=condition, encoder=encoder, chosen=chosen, grid=grid, validation_bits=bits)


def fit_fold(prepared, tempo_vocabulary, s_fields, config):
    return {c: fit_condition(prepared, c, tempo_vocabulary, s_fields, config) for c in CONDITIONS}


def test_probabilities(fit, rows):
    encoder, chosen = fit['encoder'], fit['chosen']
    own, partner = encoder.copy_indices(rows)
    return probabilities(encoder.transform(rows), chosen['w'], chosen['beta'], chosen['gamma'], own, partner)


# ---------------------------------------------------------------- selection

def select_fold(fold_id, fits):
    """v5 for one fold: selection rows for every lattice condition and rule, with every candidate's
    validation log-likelihood; the candidates are the sub-conditions' own fits (nesting)."""
    rows = []
    for condition in LATTICE:
        candidates = {condition_name(s): (s, fits[condition_name(s)]['validation_bits'])
                      for s in subconditions(condition_blocks(condition), BLOCKS)}
        results = {'max': select_max(candidates, BLOCKS), 'one_se': select_one_se(candidates, BLOCKS)}
        for rule in RULES:
            result = results[rule]
            ranked = sorted(result['validation_log_likelihood_bits'].values(), reverse=True)
            rows.append(dict(fold_id=fold_id, condition=condition, rule=rule, chosen=result['chosen'],
                             **({'best': result['best']} if rule == 'one_se' else {}),
                             validation_targets=len(next(iter(candidates.values()))[1]),
                             chosen_validation_log_likelihood_bits=result['validation_log_likelihood_bits'][result['chosen']],
                             runner_up_gap_bits=ranked[0] - ranked[1] if len(ranked) > 1 else None,
                             candidates=[dict(subcondition=name, blocks=list(blocks),
                                              validation_log_likelihood_bits=result['validation_log_likelihood_bits'][name],
                                              **(result['one_se'][name] if rule == 'one_se' else {}))
                                         for name, (blocks, _) in candidates.items()]))
    return rows


def validation_monotonicity(selection):
    """Gate 3: under the max rule every add-one-block contrast among the eight conditions is >= 0 on every
    validation day."""
    by_key = {(s['fold_id'], s['condition']): s['chosen_validation_log_likelihood_bits'] for s in selection if s['rule'] == 'max'}
    folds = sorted({s['fold_id'] for s in selection})
    contrasts = [dict(fold_id=f, pair=f'{big}-{small}', validation_contrast_bits=by_key[f, big] - by_key[f, small])
                 for f in folds for big, small in add_input_pairs(LATTICE)]
    negative = [c for c in contrasts if c['validation_contrast_bits'] < 0]
    if negative:
        raise ValueError(f'Gate 3 failed: add-one-block contrasts below zero on the validation day: {negative}')
    gaps = [s for s in selection if s['rule'] == 'max' and s['runner_up_gap_bits'] is not None]
    closest = min(gaps, key=lambda s: s['runner_up_gap_bits'])
    return dict(pairs_checked=len(contrasts), min_add_block_validation_contrast_bits=min(c['validation_contrast_bits'] for c in contrasts),
                closest_runner_up=dict(fold_id=closest['fold_id'], condition=closest['condition'], gap_bits=closest['runner_up_gap_bits']))


def model_predictions(selection, own_fit_predictions):
    """Each model's test predictions: the own fits; v5 and v5 one-SE rebuilt from the selection rows (the floor
    keeps its own fit)."""
    floor = [p for p in own_fit_predictions if p['condition'] == FLOOR]
    return {'own_fit': own_fit_predictions,
            'v5': rebuild_predictions(selection, own_fit_predictions, 'max') + floor,
            'v5_one_se': rebuild_predictions(selection, own_fit_predictions, 'one_se') + floor}


# ---------------------------------------------------------------- evaluation

def score(predictions, targets, tempo_vocabulary):
    """Per-coda log-likelihoods in bits (evaluation.log_likelihood_bits) and argmax predictions (ties: sorted
    labels, as evaluation.evaluate); every condition must score every target exactly once."""
    ll, predicted = {}, {}
    for p in predictions:
        probs = p['probabilities']
        if list(probs) != tempo_vocabulary:
            raise ValueError('Prediction labels differ from the tempo classes')
        for value in probs.values():
            log_likelihood_bits(value)
        if not math.isclose(math.fsum(probs.values()), 1, rel_tol=0, abs_tol=1e-10):
            raise ValueError('Probabilities must sum to one')
        key = (p['condition'], p['coda_id'])
        if key in ll:
            raise ValueError(f'Duplicate prediction {key}')
        ll[key] = log_likelihood_bits(probs[targets[p['coda_id']]])
        predicted[key] = max(sorted(probs), key=probs.get)
    if set(ll) != {(c, coda_id) for c in CONDITIONS for coda_id in targets}:
        raise ValueError('Predictions do not cover every condition and target exactly once')
    return ll, predicted


def contrast_stats(ll, big, small, dates, bootstrap):
    """Paired per-coda differences: day-bootstrap summary, days at exactly zero, two-sided sign-flip p."""
    values = {c: ll[big, c] - ll[small, c] for c in sorted(dates)}
    days = defaultdict(list)
    for coda_id, value in values.items():
        days[dates[coda_id]].append(value)
    stats = day_bootstrap_bits(days, bootstrap['replicates'], bootstrap['seed'], bootstrap['confidence'])
    sums, counts = day_sums(values, dates)
    stats.update(full=big, reduced=small, zero_days=sum(s == 0 for s in sums.values()),
                 sign_flip_p_two_sided=sign_flip_p(sums, counts, 'two-sided'))
    return stats


def stop_rules(ll, dates, checks, alpha):
    """DEC-23's rule on the 13 add-one-block checks, and DEC-16's old rule (CI entirely below zero)."""
    new = dec23_stop_rules({name: {c: ll[s['full'], c] - ll[s['reduced'], c] for c in dates} for name, s in checks.items()},
                           dates, alpha)
    for row in new['checks']:
        stats = checks[row['check']]
        if not math.isclose(row['mean_bits_per_coda'], stats['mean_bits_per_coda'], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{row['check']}: sign-flip input differs from the evaluated contrast")
        row.update(ci_bits_per_coda=stats['ci_bits_per_coda'], unweighted_mean_bits_per_coda=stats['unweighted_mean_bits_per_coda'],
                   positive_days=stats['positive_days'], zero_days=stats['zero_days'], old_rule_fires=stats['ci_bits_per_coda'][1] < 0)
    old = [f"{r['check']}: contrast CI entirely below zero {format_ci(r['ci_bits_per_coda'])}" for r in new['checks'] if r['old_rule_fires']]
    return dict(old=old, new=new)


def turn_category_breakdown(ll, dates, categories):
    """DEC-24's descriptive breakdown of the four contrasts by the target's turn category; no tests."""
    out = {}
    for name, (big, small) in CONTRASTS.items():
        out[name] = {}
        for category in TURN_CATEGORIES:
            codas = sorted(c for c in dates if categories[c] == category)
            if not codas:
                raise ValueError(f'No targets in turn category {category}')
            sums, counts = day_sums({c: ll[big, c] - ll[small, c] for c in codas}, dates)
            out[name][category] = dict(codas=len(codas), mean_bits_per_coda=math.fsum(sums.values()) / len(codas),
                                       days=len(sums), positive_days=sum(s > 0 for s in sums.values()),
                                       zero_days=sum(s == 0 for s in sums.values()))
    return out


def evaluate_model(predictions, targets, dates, categories, tempo_vocabulary, bootstrap, alpha):
    ll, predicted = score(predictions, targets, tempo_vocabulary)
    codas = sorted(targets)
    metrics = {c: dict(classification_metrics([targets[i] for i in codas], [predicted[c, i] for i in codas]),
                       mean_log_likelihood_bits=math.fsum(ll[c, i] for i in codas) / len(codas)) for c in CONDITIONS}
    contrasts = {name: contrast_stats(ll, big, small, dates, bootstrap) for name, (big, small) in CONTRASTS.items()}
    checks = {check_name(big, small): contrast_stats(ll, big, small, dates, bootstrap) for big, small in stop_family()}
    return dict(metrics=metrics, contrasts=contrasts, checks=checks, stop_rules=stop_rules(ll, dates, checks, alpha),
                turn_category=turn_category_breakdown(ll, dates, categories)), ll, predicted


# ---------------------------------------------------------------- run

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_targets(manifest_path, root):
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if sha256(source) != manifest['input_sha256']:
        raise ValueError(f'Stale fold manifest {manifest_path}')
    return manifest, source, read_jsonl(source)


def fit_run(spec, loaded, tempo_vocabulary, config):
    """Fits, selection and gate 3 for one run; training and validation days only."""
    manifest, coarse_manifest = loaded['manifest'], loaded['coarse_manifest']
    coarse_folds = {f['fold_id']: f for f in coarse_manifest['folds']}
    prepared, fits, selection = [], {}, []
    for fold in manifest['folds']:
        p = prepare_factor_fold(loaded['rows'], loaded['labels'], fold, coarse_folds[fold['fold_id']],
                                manifest['preparation_configuration'], loaded['s_fields'])
        prepared.append(p)
        fits[fold['fold_id']] = fit_fold(p, tempo_vocabulary, loaded['s_fields'], config)
        selection += select_fold(fold['fold_id'], fits[fold['fold_id']])
        print(f"{spec['name']}: fitted {fold['fold_id']}", flush=True)
    return prepared, fits, selection, validation_monotonicity(selection)


def fit_rows(spec, prepared, fits):
    """models.jsonl and l2_selection.jsonl rows."""
    models, selections = [], []
    for p in prepared:
        fold_id = p['fold']['fold_id']
        for condition in CONDITIONS:
            fit = fits[fold_id][condition]
            chosen = fit['chosen']
            models.append(dict(fold_id=fold_id, run=spec['name'], **fit['encoder'].record(), weights=chosen['w'].tolist(),
                               beta=None if chosen['beta'] is None else float(chosen['beta']),
                               gamma=None if chosen['gamma'] is None else float(chosen['gamma']),
                               chosen_l2=chosen['l2'], chosen_l2_s=chosen['l2_s'], convergence=chosen['convergence']))
            selections.append(dict(fold_id=fold_id, condition=condition, chosen_l2=chosen['l2'], chosen_l2_s=chosen['l2_s'],
                                   grid=[dict(l2=g['l2'], l2_s=g['l2_s'], nonconverged=g['nonconverged']) if 'nonconverged' in g else
                                         dict(l2=g['l2'], l2_s=g['l2_s'], validation_log_likelihood_bits=g['validation_log_likelihood_bits'],
                                              convergence=g['convergence']) for g in fit['grid']]))
    return models, selections


def fit_summaries(models, selections, l2_grid):
    """Chosen penalties (with grid-edge counts), copy-weight ranges and non-converged candidates per condition."""
    edges = (min(l2_grid), max(l2_grid))
    out = {}
    for condition in CONDITIONS:
        ms = [m for m in models if m['condition'] == condition]
        rows = [s for s in selections if s['condition'] == condition]
        history = [m['chosen_l2'] for m in ms]
        s_values = [m['chosen_l2_s'] for m in ms if m['chosen_l2_s'] is not None]
        betas = [m['beta'] for m in ms if m['beta'] is not None]
        gammas = [m['gamma'] for m in ms if m['gamma'] is not None]
        out[condition] = dict(history_l2=sorted(history), history_l2_at_edge=sum(v in edges for v in history),
                              s_l2=sorted(s_values) if s_values else None,
                              s_l2_at_edge=sum(v in edges for v in s_values) if s_values else None,
                              beta_range=[min(betas), max(betas)] if betas else None,
                              gamma_range=[min(gammas), max(gammas)] if gammas else None,
                              candidates=sum(len(s['grid']) for s in rows),
                              nonconverged=sum('nonconverged' in g for s in rows for g in s['grid']))
    return out


def run(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    tempo_vocabulary = list(config['tempo_classes'])
    if not tempo_vocabulary or tempo_vocabulary != sorted(set(tempo_vocabulary)):
        raise ValueError('tempo_classes must be sorted and without repeats')
    if config['governing_model'] != GOVERNING_MODEL:
        raise ValueError(f'DEC-24 governs by DEC-23 on {GOVERNING_MODEL}')
    bootstrap = dict(replicates=config['bootstrap_replicates'], seed=config['bootstrap_seed'], confidence=config['confidence'])
    reference_path = root / config['reference_manifest']
    reference = read_json(reference_path)
    paths = {path, reference_path}

    # Gates 1 and 2 (labels, folds), before any fit.
    loaded = []
    for spec in config['runs']:
        manifest_path, coarse_path = root / spec['manifest'], root / spec['pattern_manifest']
        manifest, source, rows = load_targets(manifest_path, root)
        coarse_manifest, coarse_source, coarse_rows = load_targets(coarse_path, root)
        s_fields = manifest['input_conditions']['S']
        check_folds(manifest, coarse_manifest, reference)
        labels = factor_labels(rows, coarse_rows, tempo_vocabulary, s_fields)
        paths |= {manifest_path, source, coarse_path, coarse_source}
        loaded.append(dict(manifest=manifest, coarse_manifest=coarse_manifest, rows=rows, labels=labels, s_fields=s_fields))
    tempos = [{c: t for c, (_, t) in item['labels'].items()} for item in loaded]
    if any(t != tempos[0] for t in tempos[1:]):
        raise ValueError('Tempo classes differ between runs')

    # Fits, selection and gate 3 for every run, on training and validation days only.
    fitted = [fit_run(spec, item, tempo_vocabulary, config) for spec, item in zip(config['runs'], loaded)]

    # Test days.
    input_sha256 = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}
    outputs = []
    for spec, item, (prepared, fits, selection, monotonicity) in zip(config['runs'], loaded, fitted):
        targets, dates, categories, own_fit_predictions = {}, {}, {}, []
        for p in prepared:
            fold_id = p['fold']['fold_id']
            for r in p['test']:
                targets[r['coda_id']], dates[r['coda_id']], categories[r['coda_id']] = r['tempo'], r['recording_date'], r['turn_category']
            for condition in CONDITIONS:
                probs = test_probabilities(fits[fold_id][condition], p['test'])
                own_fit_predictions += [dict(coda_id=r['coda_id'], fold_id=fold_id, condition=condition,
                                             probabilities=dict(zip(tempo_vocabulary, map(float, row))))
                                        for r, row in zip(p['test'], probs)]
        results, scored = {}, []
        for model, predictions in model_predictions(selection, own_fit_predictions).items():
            results[model], ll, predicted = evaluate_model(predictions, targets, dates, categories, tempo_vocabulary,
                                                           bootstrap, config['alpha'])
            scored += [dict(model=model, condition=c, coda_id=i, recording_date=dates[i], turn_category=categories[i],
                            target=targets[i], predicted=predicted[c, i], log_likelihood_bits=ll[c, i])
                       for c in CONDITIONS for i in sorted(targets)]
        # The own condition has one candidate and the floor none: all three models must agree on them exactly.
        for condition in ('own', FLOOR):
            if len({json.dumps(results[m]['metrics'][condition], sort_keys=True) for m in results}) != 1:
                raise ValueError(f'{condition} differs between models')
        models, selections = fit_rows(spec, prepared, fits)
        counts = {rule: {c: dict(Counter(s['chosen'] for s in selection if s['condition'] == c and s['rule'] == rule))
                         for c in LATTICE} for rule in RULES}
        summary = dict(
            units='bits per coda of the tempo class', run=spec['name'], role=spec['role'], pattern_label=spec['pattern_label'],
            model=('tempo factor (DEC-24): logistic v4 fitting per condition, v5 selection among sub-conditions (DEC-23)'),
            comparison_targets=len(targets), tempo_classes=tempo_vocabulary, conditions=CONDITIONS, blocks=list(BLOCKS),
            contrast_definitions=CONTRASTS, stop_family=[check_name(b, s) for b, s in stop_family()],
            target_counts=dict(sorted(Counter(targets.values()).items())),
            turn_category_counts=dict(sorted(Counter(categories.values()).items())),
            gates=dict(labels='pattern equals the coarse inventory label and tempo a configured class for every coda; '
                              'tempo equal across runs',
                       folds='folds equal the coarse inventory and reference manifests; pattern vocabulary equals the '
                             'coarse inventory per fold',
                       pattern_vocabulary_sizes={p['fold']['fold_id']: len(p['pattern_policy']['vocabulary']) for p in prepared},
                       validation_monotonicity=monotonicity),
            fits=fit_summaries(models, selections, config['l2_grid']),
            selection_counts=counts,
            models=results,
            governing_stop_rule=dict(model=GOVERNING_MODEL, rule='DEC-23', fired=results[GOVERNING_MODEL]['stop_rules']['new']['fired']),
            bootstrap=dict(bootstrap, method='percentile; days resampled with replacement; token-weighted mean'),
            configuration=config, run_configuration=spec,
            versions={'numpy': np.__version__, 'scipy': scipy.__version__},
            environment={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            input_sha256=input_sha256)
        outputs.append(dict(spec=spec, summary=summary, files=dict(predictions=own_fit_predictions, models=models,
                                                                   selections=selections, selection=selection, scored=scored)))

    for out in outputs:
        directory = root / out['spec']['output_directory']
        (directory / 'evaluation').mkdir(parents=True, exist_ok=True)
        files = out['files']
        write_jsonl(directory / 'predictions.jsonl', files['predictions'])
        write_jsonl(directory / 'models.jsonl', files['models'])
        write_jsonl(directory / 'l2_selection.jsonl', files['selections'])
        write_jsonl(directory / 'selection.jsonl', files['selection'])
        write_jsonl(directory / 'evaluation' / 'scored_predictions.jsonl', files['scored'])
        write_text(directory / 'evaluation' / 'summary.json', json.dumps(out['summary'], indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(config, [o['summary'] for o in outputs])) + '\n')
    return [o['summary'] for o in outputs]


# ---------------------------------------------------------------- report

def _bits(value):
    return f'{value:.5f}'


def _p(value):
    return f'{value:.4g}'


def _days(stats):
    return f"{stats['positive_days']}/{stats['days']}" + (f" ({stats['zero_days']} at 0)" if stats['zero_days'] else '')


def _pattern(stats):
    low, high = stats['ci_bits_per_coda']
    sign = '+' if stats['mean_bits_per_coda'] > 0 else '−' if stats['mean_bits_per_coda'] < 0 else '0'
    return f"{sign}, {'CI > 0' if low > 0 else 'CI < 0' if high < 0 else 'CI spans 0'}"


def contrast_label(name, pattern):
    return {'partner_tempo': 'partner tempo alone', 'partner_pattern': f'partner {pattern} alone',
            'tempo_beyond_pattern': f'tempo beyond {pattern}', 'pattern_beyond_tempo': f'{pattern} beyond tempo'}[name]


def block_label(block, pattern):
    return {'T': 'partner tempo', 'P': f'partner {pattern}', 'S': 'S', 'own': 'own history'}[block]


def _check_label(name, pattern):
    big, small = name.split(':')[1].split('-')
    if small == FLOOR:
        return f'adding own history to the floor ({big} − {small})'
    added = next(b for b in condition_blocks(big) if b not in condition_blocks(small))
    return f'adding {block_label(added, pattern)} to {small} ({big} − {small})'


def _range(values):
    return 'n/a' if values is None else f'{values[0]:.3f} to {values[1]:.3f}'


def _l2(values, at_edge, folds):
    if values is None:
        return 'n/a'
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return f'{values[0]:g} / {median:g} / {values[-1]:g} ({at_edge}/{folds} at an edge)'


def report_lines(config, summaries):
    primary = summaries[0]
    folds = len(primary['gates']['pattern_vocabulary_sizes'])
    replicates = primary['bootstrap']['replicates']
    lines = [f"# {config['report_title']}", '',
             "Does a partner's tempo or a partner's pattern predict a coda's tempo? DEC-24 (PLAN `[R2-8]`) models the chain "
             "rule's second factor directly: the target is the coda's tempo class "
             f"({', '.join(primary['tempo_classes'])}: the KDE classes of DEC-22), given the coda's own pattern, which is always "
             "present. The primary run uses Sharma rhythms as patterns (inventory D's labels); the secondary run uses "
             "`guy_coda_type` (inventory E's). Rhythm is normalised by duration, so the partner's rhythm and tempo are "
             "separable by construction; named types partly encode tempo. Every condition predicts the same five-class "
             f"target on the same {primary['comparison_targets']:,} comparison targets, with RQ1's {folds} leave-one-day-out "
             "folds and UNK policy, so every contrast is a difference in bits of the same target and the conditions nest. "
             "All log-likelihoods are in bits per coda of the tempo class.", '',
             '| Condition | Inputs besides the coda\'s own pattern |', '|---|---|',
             f'| {FLOOR} | none (the floor) |',
             '| own | own history: the whale\'s last 3 tempo classes and last 3 patterns, one-hot by lag, and a shared '
             'own-tempo copy weight |']
    for condition in LATTICE[1:]:
        added = ' + '.join({'T': "partner tempo (last 3 classes and a partner-tempo copy weight)",
                            'P': "partner pattern (last 3 patterns)", 'S': 'S'}[b] for b in condition_blocks(condition))
        lines.append(f'| {condition} | own history + {added} |')
    lines += ['', "DEC-24's base is own_S; +tempo is own_T_S, +pattern own_P_S, both own_T_P_S. Partner codas are those completed "
              "before the target's onset (`[D5]`). Tempo classes have no UNK; patterns follow RQ1's UNK policy per fold.", '',
              "**Fitting.** Logistic v4's procedure (`logistic.grid_search_l2`): every non-S column, both copy weights and the "
              "intercept share one penalty, S has its own, both chosen per fold and condition on the validation day from "
              f"{config['l2_grid']} (ties go to the larger value); the floor has one penalty. Candidates that fail the "
              "convergence check are left out of the selection and counted (`skip_nonconverged_candidates`, as in every RQ2 "
              "run). **Selection (v5, DEC-23).** For each fold and condition the candidates are the fits of every "
              "sub-condition (partner tempo, partner pattern and S each on or absent; own history always present); the "
              "highest validation-day log-likelihood wins (exact ties: fewer blocks). DEC-21's one-SE rule is the "
              "sensitivity, and the own fits (each condition's own fit, no absent candidates) are reported beside v5. "
              "DEC-23's stop rule on v5 governs each run.", '']

    lines += ['## Checks before any test-day number', '']
    for s in summaries:
        g = s['gates']
        mono = g['validation_monotonicity']
        sizes = sorted(set(g['pattern_vocabulary_sizes'].values()))
        fits = s['fits']
        candidates, nonconverged = sum(f['candidates'] for f in fits.values()), sum(f['nonconverged'] for f in fits.values())
        lines += [f"- **{s['run']} ({s['role']})**: every pattern equals the coarse inventory's label and every tempo is a "
                  "configured class (equal across runs); the folds equal the coarse inventory's and RQ1's; each fold's "
                  f"{s['pattern_label']} vocabulary (UNK included: {sizes[0]}–{sizes[-1]} labels) equals the coarse "
                  f"inventory's. Under the max rule all {mono['pairs_checked']} add-one-block contrasts (12 pairs × {folds} "
                  f"validation days) are ≥ 0 on the validation day (smallest {mono['min_add_block_validation_contrast_bits']:.3g} "
                  f"bits). {nonconverged} of {candidates:,} penalty candidates failed the convergence check."]
    lines.append('')

    lines += ['## Stop rules', '',
              "DEC-23's rule on this family's 13 add-one-block contrasts (the 12 pairs among the eight conditions, and own "
              "history added to the floor): an exact one-sided sign-flip test per check on the day sums of per-coda "
              f"differences (alternative: mean below zero; all 2^{folds} sign assignments; statistic: token-weighted mean), "
              f"Holm's correction across the 13, firing at an adjusted p < {config['alpha']}. DEC-16's old rule (day-bootstrap "
              'CI entirely below zero) is shown next to it.', '',
              '| Run | Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |', '|---|---|---|---|']
    for s in summaries:
        for model, label in MODELS:
            rules = s['models'][model]['stop_rules']
            new = '; '.join(f"{r['check']} (Holm p {_p(r['holm_p'])})" for r in rules['new']['checks'] if r['fires']) or 'none fired'
            lines.append(f"| {s['run']} | {label} | {new} | {'; '.join(rules['old']) or 'none fired'} |")
    lines.append('')
    for s in summaries:
        fired = s['governing_stop_rule']['fired']
        lines.append(f"- Governing status, {s['run']} (DEC-23's rule on v5 (max)): {', '.join(fired) or 'none fired'}."
                     + (' **Stop: no conclusion is drawn from this run; nothing was changed after seeing it.**' if fired else ''))
    lines.append('')

    lines += ['## The four contrasts (DEC-24)', '',
              'Paired per-coda differences in bits of the tempo class; positive days, with the days whose difference is exactly '
              f'zero; day-bootstrap 95% CI ({replicates:,} replicates); exact two-sided sign-flip p on the day sums (DEC-23).', '']
    for s in summaries:
        lines += [f"### {s['run']} ({s['role']}: {s['pattern_label']})", '',
                  '| Contrast | Conditions | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |',
                  '|---|---|---|---:|---:|---|---:|---:|']
        for name, (big, small) in CONTRASTS.items():
            for model, label in MODELS:
                c = s['models'][model]['contrasts'][name]
                lines.append(f"| {contrast_label(name, s['pattern_label'])} | {big} − {small} | {label} | {_bits(c['mean_bits_per_coda'])} | "
                             f"{_bits(c['unweighted_mean_bits_per_coda'])} | {format_ci(c['ci_bits_per_coda'])} | {_days(c)} | "
                             f"{_p(c['sign_flip_p_two_sided'])} |")
        lines.append('')
    lines += ['Sign of each token-weighted mean and where its CI lies, v5 (max):', '',
              '| Contrast | ' + ' | '.join(s['run'] for s in summaries) + ' |', '|---|' + '---|' * len(summaries)]
    for name in CONTRASTS:
        lines.append(f"| {contrast_label(name, 'pattern')} | "
                     + ' | '.join(_pattern(s['models']['v5']['contrasts'][name]) for s in summaries) + ' |')
    lines += ['', "Reading (DEC-24): tempo beyond the pattern > 0 with the pattern beyond tempo ≈ 0 reads as tempo matching; "
              "the pattern beyond tempo > 0 reads as the partner's pattern carrying information about the coda's speed beyond "
              "the partner's own speed (for instance through a shared behavioural context), with no claim about meaning. "
              "Nulls read \"not detectable at this data size\" (14 days). A run whose governing stop rule fires supports no "
              "reading.", '']

    lines += ['## By turn category (v5, descriptive)', '',
              "The four contrasts on the targets of each turn category; token-weighted means, no CIs and no tests (DEC-24).", '',
              '| Run | Contrast | Turn category | Targets | Token-weighted mean | Positive days |', '|---|---|---|---:|---:|---:|']
    for s in summaries:
        for name in CONTRASTS:
            for category in TURN_CATEGORIES:
                b = s['models']['v5']['turn_category'][name][category]
                lines.append(f"| {s['run']} | {contrast_label(name, s['pattern_label'])} | {category} | {b['codas']} | "
                             f"{_bits(b['mean_bits_per_coda'])} | {_days(b)} |")
    lines.append('')

    lines += ['## Lattice', '',
              'Mean log-likelihood in bits per coda of the tempo class, accuracy and macro-F1. Within a run all conditions '
              'predict the same target; the two runs condition on different patterns.', '']
    for s in summaries:
        lines += [f"### {s['run']}", '',
                  '| Condition | Own-fit LL | v5 LL | v5 one-SE LL | Own-fit accuracy | v5 accuracy | Own-fit macro-F1 | v5 macro-F1 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for c in CONDITIONS:
            a, b, o = (s['models'][m]['metrics'][c] for m in ('own_fit', 'v5', 'v5_one_se'))
            lines.append(f"| {c} | {_bits(a['mean_log_likelihood_bits'])} | {_bits(b['mean_log_likelihood_bits'])} | "
                         f"{_bits(o['mean_log_likelihood_bits'])} | {a['accuracy']:.3%} | {b['accuracy']:.3%} | "
                         f"{a['macro_f1']:.4f} | {b['macro_f1']:.4f} |")
        counts = ', '.join(f'{t} {n}' for t, n in s['target_counts'].items())
        lines += ['', f"Test targets per tempo class: {counts}.", '']

    subs = LATTICE
    lines += ['## Sub-conditions chosen', '',
              f"Folds (of {folds}) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses; a dash "
              "marks a sub-condition that is not a candidate.", '']
    for s in summaries:
        lines += [f"### {s['run']}", '', '| Condition | ' + ' | '.join(subs) + ' |', '|---|' + '---:|' * len(subs)]
        for c in LATTICE:
            cells = ['—' if not set(condition_blocks(sub)) <= set(condition_blocks(c)) else
                     f"{s['selection_counts']['max'][c].get(sub, 0)} ({s['selection_counts']['one_se'][c].get(sub, 0)})" for sub in subs]
            lines.append(f'| {c} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    lines += ['## Penalties and copy weights (own fits)', '',
              'Chosen penalties over the folds (min / median / max, and how often at an edge of the grid), the ranges of the '
              'own-tempo (β) and partner-tempo (γ) copy weights, and penalty candidates that failed the convergence check.', '']
    for s in summaries:
        lines += [f"### {s['run']}", '', '| Condition | History L2 | S L2 | β | γ | Not converged |', '|---|---|---|---|---|---:|']
        for c in CONDITIONS:
            f = s['fits'][c]
            lines.append(f"| {c} | {_l2(f['history_l2'], f['history_l2_at_edge'], folds)} | "
                         f"{_l2(f['s_l2'], f['s_l2_at_edge'], folds)} | {_range(f['beta_range'])} | {_range(f['gamma_range'])} | "
                         f"{f['nonconverged']}/{f['candidates']} |")
        lines.append('')

    lines += ['## The thirteen stop-rule checks', '',
              'Per-coda differences: token-weighted mean, day-bootstrap CI, positive days (with days at exactly zero), DEC-16 '
              '(CI entirely below zero), one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across '
              'the 13, DEC-23 (adjusted p below alpha).']
    for s in summaries:
        for model, label in MODELS:
            lines += ['', f"### {s['run']}, {label}", '',
                      '| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |',
                      '|---|---:|---|---:|---|---:|---:|---|']
            for r in s['models'][model]['stop_rules']['new']['checks']:
                lines.append(f"| {_check_label(r['check'], s['pattern_label'])} | {_bits(r['mean_bits_per_coda'])} | "
                             f"{format_ci(r['ci_bits_per_coda'])} | {_days(r)} | {'fires' if r['old_rule_fires'] else '—'} | "
                             f"{_p(r['sign_flip_p'])} | {_p(r['holm_p'])} | {'fires' if r['fires'] else '—'} |")

    lines += ['', '## Notes', '',
              f'- Day-bootstrap CIs are approximate with {folds} days, and limited data and capacity push contrasts toward zero: a '
              'CI that spans zero may mean too little data (PLAN §9).',
              '- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly '
              'favours keeping S; the one-SE rule counters this (DEC-23).',
              '- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).',
              '- Claims concern dyadic exchanges only; this report establishes no grammar, causal mechanism or meaning.',
              '- Per fold and condition: the chosen fits in `models.jsonl`, every penalty candidate in `l2_selection.jsonl`, the '
              'v5 candidates and choices in `selection.jsonl`, the own fits\' test predictions in `predictions.jsonl` (v5\'s are '
              'rebuilt from them with `logistic_v5.rebuild_predictions`), per-coda log-likelihoods of every model in '
              '`evaluation/scored_predictions.jsonl`; output directories: '
              + ', '.join(f"`{s['run_configuration']['output_directory']}/`" for s in summaries) + '.', '',
              '## Inputs', '', 'SHA-256 of every input:', '']
    lines += [f'- `{p}`: `{h}`' for p, h in primary['input_sha256'].items()]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq2_partner_tempo.json')
    for summary in run(parser.parse_args().config):
        for model, name in MODELS:
            rules = summary['models'][model]['stop_rules']
            print(f"{summary['run']}, {name}: DEC-23 rule fired for {rules['new']['fired'] or 'none'}; DEC-16 rule: {rules['old'] or 'none'}")
        governing = summary['governing_stop_rule']
        if governing['fired']:
            print(f"STOP ({summary['run']}): DEC-23's rule fired on {governing['model']}: {', '.join(governing['fired'])}. See the report.")


if __name__ == '__main__':
    main()
