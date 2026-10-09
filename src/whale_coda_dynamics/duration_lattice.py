"""Duration lattice, PLAN [P6] (DEC-30): RQ1's eight conditions with a type-free target, the coda's log duration.

The target is y = ln d, d = offset_s - onset_s, for RQ1's 1,547 comparison targets, with RQ1's 14 folds, validation
days, exchanges and S ([D5]). Inputs:
- always present, in every condition (the floor, `none`, has nothing else): ln n, n = the target's `nClicks` in its
  raw row (`source_row_index`), and one-hots for the click counts with at least `count_one_hot_min_training_targets`
  training targets in the fold; variant T adds the target's `guy_coda_type` one-hot over the fold vocabulary
  (labels and UNK per [M6]); variant F does not;
- A, B: for lags 1-3 of the whale's own (`A_history_ids`) and the partner's (`B_history_ids`) completed codas, the
  coda's ln d and ln n, each standardised on its non-missing training values and 0 when missing, and a missing
  indicator per lag; columns constant on the training rows are dropped;
- S: logistic v4's S columns (logistic.Encoder on the manifest's S fields: turn-category one-hots, gap and overlap
  elapsed raw and log1p, standardised on training rows), without its intercept.
History inputs are type-free in both variants.

Model: ridge regression on y standardised on the fold's training targets. Objective: mean squared error over the
training rows + lambda_h ||w_A, w_B||^2 + lambda_S ||w_S||^2, with a fixed `always_present_penalty` on the always-present
weights and none on the intercept; solved in closed form. Penalties come from logistic v4's grid, chosen on the
validation day by mean held-out log density, ties going to the larger value on each axis (logistic.grid_search_l2's
rule); an axis is searched only when its block has columns (the floor: no search; S: lambda_S; A, B, A_B: lambda_h;
A_S, B_S, A_B_S: every pair). sigma^2 = the mean squared training residual on the original scale. A coda's score is
evaluation.gaussian_log_density_bits(y, y_hat, sigma^2), in bits per coda; contrasts are differences of scores, so the
unit of y cancels. The copy-own-last baseline predicts the own last coda's ln d with the training mean squared
difference as variance.

Stages (--stage): checks (DEC-30's data checks against the independently computed values in the config; no models),
fit (variants F and T, every condition and fold, and the baseline), calibrate (a planted partner effect in F,
partner contrast only), report (reads only the other stages' outputs and regenerates byte-identically). Each stage
records the SHA-256 of exactly the files the config lists for it (`input_files`) and raises if it read others.
Fitting runs in one process with one BLAS thread.
"""
import argparse
import json
import math
import statistics
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np
import scipy
import scipy.linalg

from .evaluation import day_bootstrap_bits, gaussian_log_density_bits, holm, sign_flip_p
from .history_length import environment, require_current, require_one_blas_thread
from .logistic import Encoder, format_ci
from .logistic_v5 import A_CONDITIONS, add_input_pairs, condition_blocks, day_sums, dec23_stop_rules, select_max, subconditions
from .preprocessing.io_utils import read_csv_rows, read_json, read_jsonl, write_jsonl, write_text
from .residual import _hashes, _load, prepare_folds

BLOCKS = ('A', 'B', 'S')
STREAMS = (('A', 'A_history_ids'), ('B', 'B_history_ids'))
KINDS = ('log_duration', 'log_clicks')
SPLITS = ('train', 'validation', 'test')
CONTRASTS = {'own': ('A_B_S', 'B_S'), 'partner': ('A_B_S', 'A_S'), 'timing': ('A_B_S', 'A_B')}
CONTRAST_LABELS = {'own': 'own history', 'partner': 'partner', 'timing': 'timing'}
TURN_CATEGORIES = ('self_continuation', 'switch', 'overlap')
COPY = 'copy_own_last'
STAGES = ('checks', 'fit', 'calibrate', 'report')
STAGE_FILES = {'checks': ('checks/summary.json',),
               'fit': ('fit/summary.json', 'fit/l2_selection.jsonl', 'fit/models.jsonl', 'fit/scored.jsonl'),
               'calibrate': ('calibration/summary.json', 'calibration/l2_selection.jsonl', 'calibration/scored.jsonl')}
CONDITION_INPUTS = {'none': 'always-present inputs only (the floor)', 'S': 'S', 'A': 'A', 'B': 'B', 'A_B': 'A, B',
                    'A_S': 'A, S', 'B_S': 'B, S', 'A_B_S': 'A, B, S'}


# ---------------------------------------------------------------- configuration and provenance

def load(config_path):
    path, root, config, manifest, manifest_path, source = _load(config_path)
    paths = {k: root / config[k] for k in ('raw_source', 'logistic_config', 'evaluation_config')}
    loaded = dict(path=path, root=root, config=config, manifest=manifest, manifest_path=manifest_path, source=source,
                  **{k + '_path': p for k, p in paths.items()},
                  logistic_config=read_json(paths['logistic_config']), evaluation_config=read_json(paths['evaluation_config']))
    check_config(config, manifest)
    return loaded


def check_config(config, manifest):
    conditions = manifest['input_conditions']
    lattice = {'_'.join(s) or 'none' for s in subconditions(BLOCKS)}
    if set(conditions) != lattice:
        raise ValueError(f'The manifest conditions must be the power set of {BLOCKS}')
    variants = config['variants']
    if any(set(v) != {'target_type', 'label'} or type(v['target_type']) is not bool for v in variants.values()):
        raise ValueError('Each variant needs a boolean target_type and a label')
    tested = [config['primary']] + config['secondary'] + [config['calibration']]
    if any(t['variant'] not in variants or t['contrast'] not in CONTRASTS for t in tested):
        raise ValueError('Tested contrasts must name a configured variant and one of ' + ', '.join(CONTRASTS))
    calibration = config['calibration']
    if calibration['sigma_condition'] not in conditions or any(not 0 < f for f in calibration['fractions']):
        raise ValueError('The calibration needs a lattice condition for sigma and positive fractions')
    if type(config['history_lags']) is not int or not 1 <= config['history_lags'] <= manifest['preparation_configuration']['history_limit']:
        raise ValueError('history_lags must be an integer from 1 to the history limit')
    if set(config['input_files']) != set(STAGES):
        raise ValueError(f'input_files must list the inputs of every stage: {STAGES}')


def output_path(loaded, relative):
    return loaded['root'] / loaded['config']['output_directory'] / relative


def stage_paths(loaded, stage):
    return [output_path(loaded, rel) for rel in STAGE_FILES[stage]]


def external_inputs(loaded):
    """The files load() reads, whatever the stage: this config, the fold manifest, the targets, the raw rows and the
    logistic v4 and evaluation configs (grid and bootstrap settings)."""
    return [loaded['path'], loaded['manifest_path'], loaded['source'], loaded['raw_source_path'],
            loaded['logistic_config_path'], loaded['evaluation_config_path']]


def stage_inputs(loaded, stage, read):
    """SHA-256 of the files the config lists for `stage`, in the config's order; raises unless they are exactly the
    files the stage read."""
    root = loaded['root']
    listed = [root / p for p in loaded['config']['input_files'][stage]]
    if len(set(listed)) != len(listed) or set(listed) != set(read):
        missing = sorted(p.relative_to(root).as_posix() for p in set(read) - set(listed))
        extra = sorted(p.relative_to(root).as_posix() for p in set(listed) - set(read))
        raise ValueError(f'{stage}: input_files must list exactly the files the stage reads '
                         f'(read but not listed: {missing}; listed but not read: {extra})')
    return _hashes(root, listed)


def git_state(loaded):
    """The commit and the uncommitted paths, except this run's own outputs and .claude/; None outside a repository."""
    root, config = loaded['root'], loaded['config']
    own = (config['output_directory'].rstrip('/') + '/', config['report'], '.claude/')
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root, capture_output=True,
                                text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty_paths=None)
    dirty = [line[3:] for line in status if not line[3:].startswith(own)]
    return dict(commit=commit, dirty_paths=dirty)


def require_clean_tree(loaded):
    state = git_state(loaded)
    if loaded['config']['require_clean_tree'] and (state['commit'] is None or state['dirty_paths']):
        raise ValueError(f"Committed results come from a clean tree; uncommitted: {state['dirty_paths']}")
    return state


def require_stage(loaded, stage):
    """An earlier stage's outputs must exist and be current (their recorded inputs unchanged)."""
    paths = stage_paths(loaded, stage)
    if not all(p.is_file() for p in paths):
        raise ValueError(f'The {stage} stage has not been run (run --stage {stage})')
    require_current(loaded, paths[0])
    return paths


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


# ---------------------------------------------------------------- data

def parse_clicks(text):
    value = float(text)
    if not math.isfinite(value) or not value.is_integer() or value < 1:
        raise ValueError(f'Invalid nClicks {text!r}')
    return int(value)


def read_raw(loaded):
    return read_csv_rows(loaded['raw_source_path'], encoding='utf-8-sig')[1]


def coda_lookup(rows, raw_rows):
    """coda_id -> duration (offset_s - onset_s), click count (nClicks of the raw row) and their logs, for every row.
    The raw row must be the coda's (same onset as TsTo)."""
    lookup = {}
    for r in rows:
        raw = raw_rows[r['source_row_index']]
        if float(raw['TsTo']) != r['onset_s']:
            raise ValueError(f"{r['coda_id']}: raw row {r['source_row_index']} has another onset")
        duration = r['offset_s'] - r['onset_s']
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{r['coda_id']}: duration must be positive")
        clicks = parse_clicks(raw['nClicks'])
        lookup[r['coda_id']] = dict(duration_s=duration, log_duration=math.log(duration), clicks=clicks,
                                    log_clicks=math.log(clicks))
    return lookup


def history_values(row, lookup, lags):
    """Per stream (A, B), lags 1..lags: (ln d, ln n) of that completed coda, or None when the stream is shorter."""
    return {block: [(lookup[row[field][-lag]]['log_duration'], lookup[row[field][-lag]]['log_clicks'])
                    if len(row[field]) >= lag else None for lag in range(1, lags + 1)] for block, field in STREAMS}


# ---------------------------------------------------------------- checks stage (no models)

def _pearson(a, b):
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def _residuals(values, keys):
    """Each value minus the mean of the same variable over the rows sharing its key (its coda's click count or type)."""
    groups = defaultdict(list)
    for v, k in zip(values, keys):
        groups[k].append(v)
    means = {k: math.fsum(v) / len(v) for k, v in groups.items()}
    return [v - means[k] for v, k in zip(values, keys)]


def data_checks(rows, raw_rows, lags):
    """DEC-30's data checks, from the targets file (all rows for lookups) and the raw rows; no models."""
    by_id = {r['coda_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate coda IDs')
    targets = [r for r in rows if r['comparison_eligible']]
    raw = {r['coda_id']: raw_rows[r['source_row_index']] for r in rows}
    q = dict(rows=dict(count=len(rows)), comparison_targets=dict(count=len(targets)))
    references = [(t, i) for t in targets for _, field in STREAMS for i in t[field][-lags:]]
    q['history_references'] = dict(count=len(references), unresolved=sum(i not in by_id for _, i in references),
                                   ending_after_target_onset=sum(i in by_id and by_id[i]['offset_s'] > t['onset_s']
                                                                 for t, i in references))
    q['duration_vs_raw'] = dict(max_abs=max(abs(r['offset_s'] - r['onset_s'] - float(raw[r['coda_id']]['Duration'])) for r in rows))
    q['onset_vs_raw'] = dict(max_abs=max(abs(r['onset_s'] - float(raw[r['coda_id']]['TsTo'])) for r in rows))
    mismatches = []
    for r in rows:
        clicks = parse_clicks(raw[r['coda_id']]['nClicks'])
        positive = sum(float(raw[r['coda_id']][f'ICI{k}']) > 0 for k in range(1, 41))
        if clicks != positive + 1:
            mismatches.append(dict(coda_id=r['coda_id'], source_row_index=r['source_row_index'], nClicks=clicks,
                                   positive_intervals=positive, guy_coda_type=r['Ct'],
                                   comparison_target=r['comparison_eligible']))
    q['nclicks_vs_intervals'] = dict(count=len(mismatches), rows=mismatches)
    for name, field in (('own_lags', 'A_history_ids'), ('partner_lags', 'B_history_ids')):
        q[name] = dict(counts=[sum(len(t[field]) >= k for t in targets) for k in range(1, lags + 1)])
    clicks = {r['coda_id']: parse_clicks(raw[r['coda_id']]['nClicks']) for r in rows}
    log_d = {r['coda_id']: math.log(r['offset_s'] - r['onset_s']) for r in rows}
    ids = [t['coda_id'] for t in targets]
    q['targets_by_clicks'] = dict(counts={str(n): c for n, c in sorted(Counter(clicks[i] for i in ids).items())})
    y = [log_d[i] for i in ids]
    q['target_log_duration'] = dict(mean=statistics.fmean(y), sd=statistics.pstdev(y))
    keys = {'click_count': clicks, 'type': {r['coda_id']: r['Ct'] for r in rows}}
    q['target_log_duration_residual_sd'] = {name: statistics.pstdev(_residuals(y, [key[i] for i in ids]))
                                            for name, key in keys.items()}
    lag1 = {block: [t[field][-1] for t in targets] for block, field in STREAMS}
    q['lag1_r'] = dict(own=_pearson(y, [log_d[i] for i in lag1['A']]), partner=_pearson(y, [log_d[i] for i in lag1['B']]))
    # Each variable is centred on its own mean over the 1,547 rows sharing its coda's key (the target by its own
    # click count or type, a lag-1 variable by the lag-1 coda's).
    residuals = {}
    for name, key in keys.items():
        target = _residuals(y, [key[i] for i in ids])
        own = _residuals([log_d[i] for i in lag1['A']], [key[i] for i in lag1['A']])
        partner = _residuals([log_d[i] for i in lag1['B']], [key[i] for i in lag1['B']])
        residuals[name] = (target, own, partner)
        q[f'lag1_r_within_{name}'] = dict(own=_pearson(target, own), partner=_pearson(target, partner))
    target, own, partner = residuals['type']
    for category in TURN_CATEGORIES:
        keep = [j for j, t in enumerate(targets) if t['turn_category'] == category]
        q[f'lag1_r_within_type:{category}'] = dict(n=len(keep), own=_pearson([target[j] for j in keep], [own[j] for j in keep]),
                                                   partner=_pearson([target[j] for j in keep], [partner[j] for j in keep]))
    return q


def compare_checks(quantities, expected, tolerance):
    """Every expected field against the computed one: integers, lists and dicts exactly; `max_abs_below` as an upper
    bound on `max_abs`; other numbers within `tolerance` (agreement at 3 decimals)."""
    checks = []
    for name, wanted in expected.items():
        got = quantities.get(name)
        failed, computed = [], {}
        for field, value in wanted.items():
            key = 'max_abs' if field == 'max_abs_below' else field
            if got is None or key not in got:
                failed.append(field)
                computed[field] = None
                continue
            computed[field] = got[key]
            if field == 'max_abs_below':
                ok = got[key] < value
            elif isinstance(value, (bool, list, dict)) or type(value) is int:
                ok = got[key] == value
            else:
                ok = abs(got[key] - value) <= tolerance
            if not ok:
                failed.append(field)
        checks.append(dict(name=name, expected=wanted, computed=computed, failed_fields=failed, passed=not failed))
    return checks


def run_checks(config_path):
    loaded = load(config_path)
    config, settings = loaded['config'], loaded['config']['checks']
    git = require_clean_tree(loaded)
    quantities = data_checks(read_jsonl(loaded['source']), read_raw(loaded), config['history_lags'])
    checks = compare_checks(quantities, settings['expected'], settings['tolerance'])
    passed = all(c['passed'] for c in checks)
    summary = dict(source=settings['source'], tolerance=settings['tolerance'], quantities=quantities, checks=checks,
                   passed=passed, git=git, input_sha256=stage_inputs(loaded, 'checks', external_inputs(loaded)))
    write_summary(stage_paths(loaded, 'checks')[0], summary)
    failed = [c for c in checks if not c['passed']]
    print(f'Data checks: {len(checks) - len(failed)} of {len(checks)} reproduced', flush=True)
    for c in failed:
        print(f"  DIFFERS {c['name']}: computed {c['computed']}, expected {c['expected']}", flush=True)
    if failed and settings['enforce']:
        raise ValueError(f"The data checks differ from the expected values ({stage_paths(loaded, 'checks')[0]}); stop here")
    return summary


def require_checks(loaded):
    path = require_stage(loaded, 'checks')[0]
    if loaded['config']['checks']['enforce'] and not read_json(path)['passed']:
        raise ValueError('The data checks do not reproduce the expected values; stop here')
    return path


# ---------------------------------------------------------------- encoder and ridge regression

class DurationEncoder:
    """Columns in order: always-present (ln n, click-count one-hots, and in T the target's type one-hot), A, B, S,
    then the intercept. Everything is fitted on the training rows only. `groups` gives each column's penalty group:
    'always', 'A', 'B', 'S' or 'intercept'."""

    def __init__(self, train, lookup, blocks, s_fields, settings, vocabulary=None):
        if not train or any(r['split'] != 'train' for r in train):
            raise ValueError('Encoder fitting requires training rows')
        if any(b not in BLOCKS for b in blocks):
            raise ValueError(f'Unknown block in {blocks}')
        self.blocks, self.lags = tuple(blocks), settings['history_lags']
        self.vocabulary = None if vocabulary is None else list(vocabulary)
        counts = Counter(lookup[r['coda_id']]['clicks'] for r in train)
        self.count_levels = sorted(c for c, k in counts.items() if k >= settings['count_one_hot_min_training_targets'])
        self.names = (['log_clicks'] + [f'clicks={c}' for c in self.count_levels]
                      + ([] if self.vocabulary is None else [f'type={t}' for t in self.vocabulary]))
        self.groups = ['always'] * len(self.names)
        self.history_columns, self.dropped = [], []
        for block, field in STREAMS:
            if block not in self.blocks:
                continue
            for lag in range(1, self.lags + 1):
                present = [lookup[r[field][-lag]] for r in train if len(r[field]) >= lag]
                for kind in KINDS:
                    values = [c[kind] for c in present]
                    name = f'{block}:lag{lag}:{kind}'
                    if values and float(np.std(values)) > 0:
                        self.history_columns.append(dict(name=name, block=block, field=field, lag=lag, kind=kind,
                                                         mean=float(np.mean(values)), scale=float(np.std(values))))
                    else:
                        self.dropped.append(name)
                name = f'{block}:lag{lag}:missing'
                if 0 < len(present) < len(train):
                    self.history_columns.append(dict(name=name, block=block, field=field, lag=lag, kind='missing'))
                else:
                    self.dropped.append(name)
        self.names += [c['name'] for c in self.history_columns]
        self.groups += [c['block'] for c in self.history_columns]
        self.s_fields = list(s_fields) if 'S' in self.blocks else []
        self.s_encoder = Encoder(train, [], self.s_fields) if 'S' in self.blocks else None
        if self.s_encoder is not None:
            self.names += ['S:' + n for n in self.s_encoder.names]
            self.groups += ['S'] * len(self.s_encoder.names)
        self.names.append('intercept')
        self.groups.append('intercept')
        self.has_history = any(g in ('A', 'B') for g in self.groups)
        self.has_s = 'S' in self.groups

    def transform(self, rows, lookup):
        x = []
        for r in rows:
            coda = lookup[r['coda_id']]
            values = [coda['log_clicks']] + [float(coda['clicks'] == c) for c in self.count_levels]
            if self.vocabulary is not None:
                if r['Ct'] not in self.vocabulary:
                    raise ValueError(f"Target type {r['Ct']!r} is not in the fold vocabulary")
                values += [float(r['Ct'] == t) for t in self.vocabulary]
            for c in self.history_columns:
                ids = r[c['field']]
                present = len(ids) >= c['lag']
                if c['kind'] == 'missing':
                    values.append(float(not present))
                else:
                    values.append((lookup[ids[-c['lag']]][c['kind']] - c['mean']) / c['scale'] if present else 0.0)
            x.append(values)
        n_before_s = len(self.names) - 1 - (len(self.s_encoder.names) if self.s_encoder is not None else 0)
        x = np.asarray(x, dtype=float).reshape(len(rows), n_before_s)
        if self.s_encoder is not None:
            x = np.column_stack([x, self.s_encoder.transform(rows)[:, :-1]])
        return np.column_stack([x, np.ones(len(rows))])

    def penalties(self, l2, l2_s, fixed):
        value = {'always': fixed, 'A': l2, 'B': l2, 'S': l2_s, 'intercept': 0.0}
        if (l2 is None) == self.has_history or (l2_s is None) == self.has_s:
            raise ValueError('A penalty is given exactly for each present penalty group')
        return np.array([value[g] for g in self.groups], dtype=float)

    def record(self):
        return dict(blocks=list(self.blocks), feature_names=self.names, groups=self.groups, count_levels=self.count_levels,
                    vocabulary=self.vocabulary, history_columns=self.history_columns, dropped_columns=self.dropped,
                    s_fields=self.s_fields, categories=self.s_encoder.categories if self.s_encoder else [],
                    s_mean=self.s_encoder.mean.tolist() if self.s_encoder else [],
                    s_scale=self.s_encoder.scale.tolist() if self.s_encoder else [])


def ridge_weights(x, z, penalties):
    """argmin_w mean((z - x w)^2) + sum_j penalties_j w_j^2: (x'x/n + diag(penalties)) w = x'z/n."""
    n = len(z)
    gram = x.T @ x / n
    gram[np.diag_indices_from(gram)] += penalties
    return scipy.linalg.solve(gram, x.T @ z / n, assume_a='pos')


def ridge_fit(x, y, penalties):
    """Ridge regression on y standardised on these (training) rows; sigma^2 = the mean squared training residual on
    the original scale."""
    mean, scale = float(np.mean(y)), float(np.std(y))
    if not scale > 0:
        raise ValueError('The training targets have no variance')
    w = ridge_weights(x, (y - mean) / scale, penalties)
    residual = y - (mean + scale * (x @ w))
    return dict(weights=w, y_mean=mean, y_scale=scale, variance=float(np.mean(residual ** 2)))


def predict(fit, x):
    return fit['y_mean'] + fit['y_scale'] * (x @ fit['weights'])


def log_density_bits(y, mean, variance):
    """Per-coda scores through the shared evaluation module."""
    return np.array([gaussian_log_density_bits(float(a), float(b), float(variance)) for a, b in zip(y, mean)])


def penalty_search(encoder, x_train, y_train, x_val, y_val, grid, fixed):
    """The validation-day search: each present axis over the grid; the highest mean log density wins, ties to the
    larger value on each axis (logistic.grid_search_l2's rule)."""
    candidates = []
    for l2 in (grid if encoder.has_history else [None]):
        for l2_s in (grid if encoder.has_s else [None]):
            fit = ridge_fit(x_train, y_train, encoder.penalties(l2, l2_s, fixed))
            bits = log_density_bits(y_val, predict(fit, x_val), fit['variance'])
            candidates.append(dict(fit, l2=l2, l2_s=l2_s, validation_bits=bits,
                                   validation_log_density_bits=math.fsum(bits) / len(bits)))
    chosen = max(candidates, key=lambda c: (c['validation_log_density_bits'], c['l2'] or 0.0, c['l2_s'] or 0.0))
    return chosen, candidates


def target_values(prepared, y_by_id):
    return {s: np.array([y_by_id[r['coda_id']] for r in prepared[s]], dtype=float) for s in SPLITS}


def fit_condition(prepared, lookup, y_by_id, condition, target_type, settings, s_fields, grid):
    """One fold and condition on training and validation days; the chosen fit scored on the test day."""
    vocabulary = prepared['policy']['vocabulary'] if target_type else None
    encoder = DurationEncoder(prepared['train'], lookup, condition_blocks(condition), s_fields, settings, vocabulary)
    x = {s: encoder.transform(prepared[s], lookup) for s in SPLITS}
    y = target_values(prepared, y_by_id)
    chosen, grid_rows = penalty_search(encoder, x['train'], y['train'], x['validation'], y['validation'], grid,
                                       settings['always_present_penalty'])
    prediction = predict(chosen, x['test'])
    return dict(encoder=encoder, chosen=chosen, grid=grid_rows, y_test=y['test'], prediction=prediction,
                test_bits=log_density_bits(y['test'], prediction, chosen['variance']))


def copy_baseline(prepared, lookup, y_by_id):
    """The own last coda's ln d as the prediction, with the training mean squared difference as variance."""
    y = target_values(prepared, y_by_id)
    if any(not r['A_history_ids'] for s in SPLITS for r in prepared[s]):
        raise ValueError('Every comparison target needs an own last coda')
    last = {s: np.array([lookup[r['A_history_ids'][-1]]['log_duration'] for r in prepared[s]]) for s in SPLITS}
    variance = float(np.mean((y['train'] - last['train']) ** 2))
    return dict(variance=variance, y_test=y['test'], prediction=last['test'],
                test_bits=log_density_bits(y['test'], last['test'], variance))


# ---------------------------------------------------------------- fit stage

def scored_rows(model, condition, prepared, fitted, stored):
    fold_id, sigma = prepared['fold']['fold_id'], math.sqrt(fitted['chosen']['variance'] if 'chosen' in fitted else fitted['variance'])
    out = []
    for r, y, mean, bits in zip(prepared['test'], fitted['y_test'], fitted['prediction'], fitted['test_bits']):
        s = stored[r['coda_id']]
        out.append(dict(model=model, condition=condition, fold_id=fold_id, coda_id=r['coda_id'],
                        recording_date=r['recording_date'], turn_category=s['turn_category'],
                        repeats_own_last_type=s['Ct'] == s['A_history'][-1], target_log_duration=float(y),
                        prediction=float(mean), sigma=sigma, log_density_bits=float(bits)))
    return out


def grid_record(model, fold_id, condition, chosen, grid):
    return dict(model=model, fold_id=fold_id, condition=condition, chosen_l2=chosen['l2'], chosen_l2_s=chosen['l2_s'],
                grid=[dict(l2=g['l2'], l2_s=g['l2_s'], validation_log_density_bits=g['validation_log_density_bits'],
                           variance=g['variance']) for g in grid])


def model_record(model, prepared, condition, fitted):
    chosen = fitted['chosen']
    return dict(model=model, fold_id=prepared['fold']['fold_id'], condition=condition, **fitted['encoder'].record(),
                weights=chosen['weights'].tolist(), y_mean=chosen['y_mean'], y_scale=chosen['y_scale'],
                variance=chosen['variance'], chosen_l2=chosen['l2'], chosen_l2_s=chosen['l2_s'],
                validation_log_density_bits=chosen['validation_log_density_bits'],
                validation_ids=[r['coda_id'] for r in prepared['validation']],
                validation_bits=chosen['validation_bits'].tolist())


def prepared_data(loaded):
    rows = read_jsonl(loaded['source'])
    lookup = coda_lookup(rows, read_raw(loaded))
    prepared = prepare_folds(rows, loaded['manifest'])
    return rows, lookup, prepared


def run_fit(config_path):
    require_one_blas_thread()
    loaded = load(config_path)
    config, manifest = loaded['config'], loaded['manifest']
    git = require_clean_tree(loaded)
    checks_path = require_checks(loaded)
    start = time.perf_counter()
    rows, lookup, prepared = prepared_data(loaded)
    stored = {r['coda_id']: r for r in rows}
    y_by_id = {c: v['log_duration'] for c, v in lookup.items()}
    s_fields, grid = manifest['input_conditions']['S'], loaded['logistic_config']['l2_grid']
    conditions = list(manifest['input_conditions'])
    selections, models, scored, seconds, fits = [], [], [], {}, 0
    for variant, spec in config['variants'].items():
        begin = time.perf_counter()
        for p in prepared:
            for condition in conditions:
                fitted = fit_condition(p, lookup, y_by_id, condition, spec['target_type'], config, s_fields, grid)
                fits += len(fitted['grid'])
                selections.append(grid_record(variant, p['fold']['fold_id'], condition, fitted['chosen'], fitted['grid']))
                models.append(model_record(variant, p, condition, fitted))
                scored += scored_rows(variant, condition, p, fitted, stored)
        seconds[variant] = time.perf_counter() - begin
        print(f'{variant}: {seconds[variant]:.1f} s', flush=True)
    for p in prepared:
        baseline = copy_baseline(p, lookup, y_by_id)
        models.append(dict(model=COPY, fold_id=p['fold']['fold_id'], condition=COPY, variance=baseline['variance']))
        scored += scored_rows(COPY, COPY, p, baseline, stored)
    total = time.perf_counter() - start
    metrics = defaultdict(dict)
    for r in scored:
        metrics[r['model']].setdefault(r['condition'], []).append(r['log_density_bits'])
    summary = dict(units='bits per coda (log2 Gaussian density of ln d)', comparison_targets=len({r['coda_id'] for r in scored}),
                   folds=len(prepared), variants=config['variants'], conditions=conditions,
                   fits=fits, fit_seconds=seconds, total_seconds=total,
                   mean_log_density_bits={m: {c: math.fsum(v) / len(v) for c, v in cs.items()} for m, cs in metrics.items()},
                   environment=environment(), git=git,
                   input_sha256=stage_inputs(loaded, 'fit', external_inputs(loaded) + [checks_path]))
    paths = stage_paths(loaded, 'fit')
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths[1], selections)
    write_jsonl(paths[2], models)
    write_jsonl(paths[3], scored)
    write_summary(paths[0], summary)
    print(f'Fit stage: {fits} penalty candidates, {total:.1f} s', flush=True)
    return summary


# ---------------------------------------------------------------- calibration stage

def planted_targets(prepared, lookup, y_by_id, fraction, sigma, settings, s_fields):
    """y + fraction * sigma * z_b1 on every split of the fold, z_b1 the standardised ln d of the partner's last coda
    (0 when missing): the A_B_S encoder's own B lag-1 ln d column, standardised on the training rows."""
    encoder = DurationEncoder(prepared['train'], lookup, ('A', 'B', 'S'), s_fields, settings)
    if 'B:lag1:log_duration' not in encoder.names:
        raise ValueError("The partner's last ln d is constant on the training rows; nothing to plant")
    column = encoder.names.index('B:lag1:log_duration')
    planted = {}
    for s in SPLITS:
        z = encoder.transform(prepared[s], lookup)[:, column]
        for r, value in zip(prepared[s], z):
            planted[r['coda_id']] = y_by_id[r['coda_id']] + fraction * sigma * float(value)
    return planted


def run_calibration(config_path):
    require_one_blas_thread()
    loaded = load(config_path)
    config, manifest = loaded['config'], loaded['manifest']
    git = require_clean_tree(loaded)
    checks_path = require_checks(loaded)
    fit_summary_path, _, models_path, _ = require_stage(loaded, 'fit')
    spec = config['calibration']
    target_type = config['variants'][spec['variant']]['target_type']
    sigma_models = {r['fold_id']: r for r in read_jsonl(models_path)
                    if r['model'] == spec['variant'] and r['condition'] == spec['sigma_condition']}
    start = time.perf_counter()
    rows, lookup, prepared = prepared_data(loaded)
    stored = {r['coda_id']: r for r in rows}
    y_by_id = {c: v['log_duration'] for c, v in lookup.items()}
    s_fields, grid = manifest['input_conditions']['S'], loaded['logistic_config']['l2_grid']
    if set(sigma_models) != {p['fold']['fold_id'] for p in prepared}:
        raise ValueError('The fit stage lacks a fold of the sigma condition')
    selections, scored, sigmas, fits = [], [], {}, 0
    for fraction in spec['fractions']:
        for p in prepared:
            fold_id = p['fold']['fold_id']
            sigmas[fold_id] = math.sqrt(sigma_models[fold_id]['variance'])
            planted = planted_targets(p, lookup, y_by_id, fraction, sigmas[fold_id], config, s_fields)
            for condition in CONTRASTS[spec['contrast']]:
                fitted = fit_condition(p, lookup, planted, condition, target_type, config, s_fields, grid)
                fits += len(fitted['grid'])
                selections.append(dict(grid_record(spec['variant'], fold_id, condition, fitted['chosen'], fitted['grid']),
                                       fraction=fraction))
                scored += [dict(r, fraction=fraction) for r in scored_rows(spec['variant'], condition, p, fitted, stored)]
        print(f'Calibration f = {fraction}: done', flush=True)
    total = time.perf_counter() - start
    summary = dict(configuration=spec, sigma_by_fold=sigmas, fits=fits, seconds=total, environment=environment(), git=git,
                   input_sha256=stage_inputs(loaded, 'calibrate',
                                             external_inputs(loaded) + [checks_path, fit_summary_path, models_path]))
    paths = stage_paths(loaded, 'calibrate')
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths[1], selections)
    write_jsonl(paths[2], scored)
    write_summary(paths[0], summary)
    print(f'Calibration stage: {fits} penalty candidates, {total:.1f} s', flush=True)
    return summary


# ---------------------------------------------------------------- report stage: statistics

def contrast_stats(ll, big, small, dates, boot, sign_flip=False):
    """Paired per-coda differences: day-bootstrap summary, days at exactly zero, optionally the two-sided sign-flip p."""
    values = {c: ll[big][c] - ll[small][c] for c in sorted(dates)}
    days = defaultdict(list)
    for coda_id, value in values.items():
        days[dates[coda_id]].append(value)
    stats = day_bootstrap_bits(dict(days), *boot)
    sums, counts = day_sums(values, dates)
    stats.update(full=big, reduced=small, zero_days=sum(s == 0 for s in sums.values()))
    if sign_flip:
        stats['sign_flip_p_two_sided'] = sign_flip_p(sums, counts, 'two-sided')
    return stats


def detectable(stats, p_key, alpha):
    return stats['ci_bits_per_coda'][0] > 0 and stats[p_key] < alpha


def breakdown(ll, dates, groups, labels):
    """The three RQ1 contrasts within each group of targets: token-weighted mean, targets, positive days; no tests."""
    out = {}
    for name, (big, small) in CONTRASTS.items():
        out[name] = {}
        for label in labels:
            codas = sorted(c for c in dates if groups[c] == label)
            sums, counts = day_sums({c: ll[big][c] - ll[small][c] for c in codas}, dates)
            out[name][str(label)] = dict(codas=len(codas), mean_bits_per_coda=math.fsum(sums.values()) / len(codas),
                                         days=len(sums), positive_days=sum(s > 0 for s in sums.values()))
    return out


def residual_diagnostics(rows):
    """Held-out residuals standardised by each fold's sigma: skewness, excess kurtosis, the share beyond 3 sigma and
    the mean squared standardised residual (1 for a calibrated variance)."""
    z = np.array([(r['target_log_duration'] - r['prediction']) / r['sigma'] for r in rows])
    centred = z - z.mean()
    m2 = float(np.mean(centred ** 2))
    return dict(codas=len(z), skewness=float(np.mean(centred ** 3)) / m2 ** 1.5, excess_kurtosis=float(np.mean(centred ** 4)) / m2 ** 2 - 3,
                beyond_3_sigma=int(np.sum(np.abs(z) > 3)), share_beyond_3_sigma=float(np.mean(np.abs(z) > 3)),
                mean_squared_standardised_residual=float(np.mean(z ** 2)),
                held_out_rms_residual=math.sqrt(math.fsum((r['target_log_duration'] - r['prediction']) ** 2 for r in rows) / len(rows)))


def lag1_slopes(models, variant, condition):
    """Per fold, the fitted change in predicted ln d per unit of the own (A) and partner (B) last coda's ln d."""
    out = {}
    for block in ('A', 'B'):
        if block not in condition_blocks(condition):
            continue
        name, per_fold = f'{block}:lag1:log_duration', {}
        for m in models:
            if m['model'] == variant and m['condition'] == condition:
                column = next(c for c in m['history_columns'] if c['name'] == name)
                per_fold[m['fold_id']] = m['y_scale'] * m['weights'][m['feature_names'].index(name)] / column['scale']
        values = list(per_fold.values())
        out[block] = dict(min=min(values), median=statistics.median(values), max=max(values), per_fold=per_fold)
    return out


def penalty_summary(models, variant, condition, grid):
    rows = [m for m in models if m['model'] == variant and m['condition'] == condition]
    edges = (min(grid), max(grid))
    out = dict(sigma_median=statistics.median(math.sqrt(m['variance']) for m in rows))
    for key in ('chosen_l2', 'chosen_l2_s'):
        values = [m[key] for m in rows if m[key] is not None]
        out[key] = dict(median=statistics.median(values), at_grid_edge=sum(v in edges for v in values)) if values else None
    return out


def v5_line(models, ll, scored_fold, conditions, variant):
    """DEC-23's absent candidates on these fits: per fold and condition, the sub-condition fit with the highest
    validation-day mean log density (logistic_v5.select_max: exact ties to fewer blocks, then A, B, S); its
    test-day scores become the condition's. No new fits."""
    validation = {}
    for m in models:
        if m['model'] == variant:
            validation[m['fold_id'], m['condition']] = (m['validation_ids'], np.array(m['validation_bits']))
    chosen, v5 = {}, {c: {} for c in conditions}
    for fold_id in sorted({f for f, _ in validation}):
        ids = {tuple(validation[fold_id, c][0]) for c in conditions}
        if len(ids) != 1:
            raise ValueError(f'{fold_id}: conditions were validated on different targets')
        for condition in conditions:
            candidates = {'_'.join(s) or 'none': (s, validation[fold_id, '_'.join(s) or 'none'][1])
                          for s in subconditions(condition_blocks(condition))}
            pick = select_max(candidates)['chosen']
            chosen[fold_id, condition] = pick
            for coda_id in scored_fold[fold_id]:
                v5[condition][coda_id] = ll[pick][coda_id]
    counts = {c: dict(sorted(Counter(chosen[f, c] for f, cc in chosen if cc == c).items())) for c in conditions}
    return v5, counts


def stop_rules(ll, copy, dates, boot, alpha):
    """DEC-23's rule on the 16 checks (12 add-one-input contrasts, the 4 A-conditions against the copy baseline), with
    DEC-16's (interval entirely below zero) beside it."""
    checks, stats = {}, {}
    for big, small in add_input_pairs(list(ll)):
        name = f'add_input:{big}-{small}'
        checks[name] = {c: ll[big][c] - ll[small][c] for c in dates}
        stats[name] = contrast_stats(ll, big, small, dates, boot)
    for a in A_CONDITIONS:
        name = f'{COPY}:{a}'
        checks[name] = {c: ll[a][c] - copy[c] for c in dates}
        stats[name] = contrast_stats(dict(ll, **{COPY: copy}), a, COPY, dates, boot)
    new = dec23_stop_rules(checks, dates, alpha)
    for row in new['checks']:
        s = stats[row['check']]
        if not math.isclose(row['mean_bits_per_coda'], s['mean_bits_per_coda'], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{row['check']}: sign-flip input differs from the evaluated contrast")
        row.update(ci_bits_per_coda=s['ci_bits_per_coda'], unweighted_mean_bits_per_coda=s['unweighted_mean_bits_per_coda'],
                   positive_days=s['positive_days'], old_rule_fires=s['ci_bits_per_coda'][1] < 0)
    return dict(new=new, old=[r['check'] for r in new['checks'] if r['old_rule_fires']],
                fitting_flags=[r['check'] for r in new['checks'] if r['check'].startswith('add_input:') and r['old_rule_fires']])


def nominal_bits(fraction):
    """The partner contrast's information if z_b1 (unit variance) were independent of A and S: 1/2 log2(1 + f^2)."""
    return 0.5 * math.log2(1 + fraction ** 2)


# ---------------------------------------------------------------- report stage

def run_report(config_path):
    loaded = load(config_path)
    root, config, manifest = loaded['root'], loaded['config'], loaded['manifest']
    evaluation = loaded['evaluation_config']
    boot = (evaluation['bootstrap_replicates'], evaluation['bootstrap_seed'], evaluation['confidence'])
    alpha, grid = config['alpha'], loaded['logistic_config']['l2_grid']
    require_clean_tree(loaded)
    checks_path = require_checks(loaded)
    fit_paths, calibration_paths = require_stage(loaded, 'fit'), require_stage(loaded, 'calibrate')
    read = external_inputs(loaded) + [checks_path, fit_paths[0], fit_paths[2], fit_paths[3],
                                      calibration_paths[0], calibration_paths[2]]
    checks = read_json(checks_path)
    fit_summary, models, scored = read_json(fit_paths[0]), read_jsonl(fit_paths[2]), read_jsonl(fit_paths[3])
    calibration_summary, calibration_scored = read_json(calibration_paths[0]), read_jsonl(calibration_paths[2])
    conditions = list(manifest['input_conditions'])

    ll, dates, categories, repeats, scored_fold, by_model = defaultdict(lambda: defaultdict(dict)), {}, {}, {}, defaultdict(set), defaultdict(list)
    for r in scored:
        if r['coda_id'] in ll[r['model']][r['condition']]:
            raise ValueError(f"Duplicate scored row {r['model']}, {r['condition']}, {r['coda_id']}")
        ll[r['model']][r['condition']][r['coda_id']] = r['log_density_bits']
        by_model[r['model'], r['condition']].append(r)
        for table, value in ((dates, r['recording_date']), (categories, r['turn_category']), (repeats, r['repeats_own_last_type'])):
            if table.setdefault(r['coda_id'], value) != value:
                raise ValueError(f"{r['coda_id']}: scored rows disagree on its day, turn category or repeat flag")
        scored_fold[r['fold_id']].add(r['coda_id'])
    ids = set(dates)
    expected_ids = {c for fold in manifest['folds'] for c in fold['comparison_ids']['test']}
    if ids != expected_ids:
        raise ValueError('The scored targets are not the manifest test targets')
    variants = list(config['variants'])
    for model, by_condition in ll.items():
        if set(by_condition) != (set(conditions) if model in variants else {COPY}) or any(set(v) != ids for v in by_condition.values()):
            raise ValueError(f'{model}: scored rows do not cover every condition and target exactly once')
    copy = ll[COPY][COPY]

    results = {}
    for variant in variants:
        lattice = ll[variant]
        metrics = {c: dict(mean_log_density_bits=math.fsum(lattice[c].values()) / len(ids), **penalty_summary(models, variant, c, grid),
                           residuals=residual_diagnostics(by_model[variant, c])) for c in conditions}
        add_one = {f'{big}-{small}': contrast_stats(lattice, big, small, dates, boot) for big, small in add_input_pairs(conditions)}
        rq1 = {name: contrast_stats(lattice, big, small, dates, boot, sign_flip=True) for name, (big, small) in CONTRASTS.items()}
        v5, v5_counts = v5_line(models, lattice, scored_fold, conditions, variant)
        results[variant] = dict(
            metrics=metrics, add_one_input=add_one, rq1_contrasts=rq1,
            versus_floor={c: contrast_stats(lattice, c, 'none', dates, boot) for c in conditions if c != 'none'},
            by_turn_category=breakdown(lattice, dates, categories, TURN_CATEGORIES),
            by_repeat=breakdown(lattice, dates, repeats, (True, False)),
            lag1_slopes={c: lag1_slopes(models, variant, c) for c in conditions if set(condition_blocks(c)) & {'A', 'B'}},
            stop_rules=stop_rules(lattice, copy, dates, boot, alpha),
            v5=dict(mean_log_density_bits={c: math.fsum(v5[c].values()) / len(ids) for c in conditions},
                    selection_counts=v5_counts,
                    contrasts={name: contrast_stats(v5, big, small, dates, boot, sign_flip=True) for name, (big, small) in CONTRASTS.items()}))

    tested = []
    for role, specs in (('primary', [config['primary']]), ('secondary', config['secondary'])):
        for spec in specs:
            stats = results[spec['variant']]['rq1_contrasts'][spec['contrast']]
            tested.append(dict(role=role, variant=spec['variant'], contrast=spec['contrast'], full=stats['full'], reduced=stats['reduced'],
                               mean_bits_per_coda=stats['mean_bits_per_coda'], unweighted_mean_bits_per_coda=stats['unweighted_mean_bits_per_coda'],
                               ci_bits_per_coda=stats['ci_bits_per_coda'], positive_days=stats['positive_days'], days=stats['days'],
                               sign_flip_p_two_sided=stats['sign_flip_p_two_sided']))
    secondary = [t for t in tested if t['role'] == 'secondary']
    for t, adjusted in zip(secondary, holm([t['sign_flip_p_two_sided'] for t in secondary])):
        t['holm_p'] = adjusted
    for t in tested:
        t['detectable'] = detectable(t, 'sign_flip_p_two_sided' if t['role'] == 'primary' else 'holm_p', alpha)

    real_variant = config['calibration']['variant']
    real_sigma = defaultdict(dict)
    for r in scored:
        if r['model'] == real_variant:
            real_sigma[r['condition']][r['fold_id']] = r['sigma']
    calibration = calibration_result(calibration_scored, calibration_summary, config, boot, ll[real_variant], real_sigma)
    result = dict(units='bits per coda: log2 Gaussian density of ln d (d in seconds); only differences are compared',
                  comparison_targets=len(ids), days=len({dates[c] for c in ids}), conditions=conditions, variants=config['variants'],
                  bootstrap=dict(zip(('replicates', 'seed', 'confidence'), boot)), alpha=alpha, l2_grid=grid,
                  always_present_penalty=config['always_present_penalty'], history_lags=config['history_lags'],
                  count_one_hot_min_training_targets=config['count_one_hot_min_training_targets'],
                  checks={k: checks[k] for k in ('source', 'tolerance', 'checks', 'passed')},
                  process_notes=config.get('process_notes', []),
                  nclicks_mismatch_rows=checks['quantities']['nclicks_vs_intervals']['rows'],
                  copy_own_last=dict(mean_log_density_bits=math.fsum(copy.values()) / len(ids),
                                     residuals=residual_diagnostics(by_model[COPY, COPY])),
                  results=results, tested=tested, calibration=calibration,
                  fit=dict(fits=fit_summary['fits'], fit_seconds=fit_summary['fit_seconds'], total_seconds=fit_summary['total_seconds'],
                           calibration_fits=calibration_summary['fits'], calibration_seconds=calibration_summary['seconds'],
                           environment=fit_summary['environment'], git=fit_summary['git']))
    result['reading'] = reading_lines(result, config)
    result['input_sha256'] = stage_inputs(loaded, 'report', read)
    write_summary(output_path(loaded, 'summary.json'), result)
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(result, config)) + '\n')
    return result


def calibration_result(rows, summary, config, boot, real_ll, real_sigma):
    """The planted partner contrast per f, with its paired increment over the real-data contrast and, per condition,
    the median over folds of the planted fit's training sigma over the real fit's (sqrt(1 + f^2) for the reduced
    condition if z_b1 were independent of its residual)."""
    spec = config['calibration']
    big, small = CONTRASTS[spec['contrast']]
    per_fraction = {}
    for fraction in spec['fractions']:
        ll, dates, sigma = defaultdict(dict), {}, defaultdict(dict)
        for r in rows:
            if r['fraction'] == fraction:
                ll[r['condition']][r['coda_id']] = r['log_density_bits']
                dates[r['coda_id']] = r['recording_date']
                sigma[r['condition']][r['fold_id']] = r['sigma']
        stats = contrast_stats(ll, big, small, dates, boot, sign_flip=True)
        stats.pop('day_means_bits')
        paired = {'planted': {c: ll[big][c] - ll[small][c] for c in dates}, 'real': {c: real_ll[big][c] - real_ll[small][c] for c in dates}}
        increment = contrast_stats(paired, 'planted', 'real', dates, boot)
        increment.pop('day_means_bits')
        ratios = {c: statistics.median(s / real_sigma[c][f] for f, s in sigma[c].items()) for c in (big, small)}
        per_fraction[str(fraction)] = dict(fraction=fraction, nominal_maximum_bits_per_coda=nominal_bits(fraction),
                                           detected=stats['ci_bits_per_coda'][0] > 0,
                                           interval_below_zero=stats['ci_bits_per_coda'][1] < 0, **stats,
                                           increment_over_real=increment, sigma_ratio_median=ratios,
                                           sigma_ratio_if_independent=math.sqrt(1 + fraction ** 2))
    detected = [p['fraction'] for p in per_fraction.values() if p['detected']]
    return dict(variant=spec['variant'], contrast=spec['contrast'], full=big, reduced=small,
                sigma_condition=spec['sigma_condition'], sigma_by_fold=summary['sigma_by_fold'], per_fraction=per_fraction,
                detection_bound_fraction=min(detected) if detected else None,
                failed=[p['fraction'] for p in per_fraction.values() if p['interval_below_zero']])


# ---------------------------------------------------------------- reading and report

def _f(value, digits=5):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def _p(value):
    return f'{value:.4g}'


def _days(stats):
    return f"{stats['positive_days']}/{stats['days']}" + (f" ({stats['zero_days']} at 0)" if stats.get('zero_days') else '')


def _tested_text(t):
    holm_text = f", Holm-adjusted p = {_p(t['holm_p'])}" if 'holm_p' in t else ''
    return (f"{CONTRAST_LABELS[t['contrast']]} LL({t['full'].replace('_', ',')}) − LL({t['reduced'].replace('_', ',')}) in "
            f"{t['variant']}: {t['mean_bits_per_coda']:+.4f} bits/coda (unweighted day mean {t['unweighted_mean_bits_per_coda']:+.4f}, "
            f"95% CI {format_ci(t['ci_bits_per_coda'])}, {t['positive_days']}/{t['days']} days positive, two-sided sign-flip "
            f"p = {_p(t['sign_flip_p_two_sided'])}{holm_text})")


def bound_text(calibration):
    bound = calibration['detection_bound_fraction']
    if bound is None:
        return (f"The calibration gives no bound: no planted partner effect up to f = {max(float(f) for f in calibration['per_fraction'])} "
                'had an interval above zero.')
    p = calibration['per_fraction'][str(bound)]
    smaller = [q for q in calibration['per_fraction'].values() if q['fraction'] < bound]
    missed = [q for q in calibration['per_fraction'].values() if q['fraction'] > bound and not q['detected']]
    planted = lambda q: f"f = {q['fraction']} ({q['mean_bits_per_coda']:+.3f})"  # noqa: E731
    return (f"Calibration: the smallest planted partner effect detected is f = {bound}: planted partner contrast "
            f"{p['mean_bits_per_coda']:+.3f} bits/coda (interval {format_ci(p['ci_bits_per_coda'])}), "
            f"{p['increment_over_real']['mean_bits_per_coda']:+.3f} (interval {format_ci(p['increment_over_real']['ci_bits_per_coda'])}) "
            'above the real-data contrast'
            + (f"; {', '.join(planted(q) for q in smaller)} {'was' if len(smaller) == 1 else 'were'} not detected" if smaller else
               '; it is the smallest planted effect tried')
            + (f"; larger {', '.join(planted(q) for q in missed)} not detected" if missed else '') + '.'
            + nominal_text(calibration))


def nominal_text(calibration):
    over = [q for q in calibration['per_fraction'].values()
            if q['increment_over_real']['mean_bits_per_coda'] > q['nominal_maximum_bits_per_coda']]
    if not over:
        return " Every planted increment is within DEC-30's nominal size ½·log2(1 + f²)."
    return (" DEC-30's nominal sizes, ½·log2(1 + f²), assume z_b1 independent of the reduced model's residual; here the planted "
            'increment exceeds them at f = ' + ', '.join(str(q['fraction']) for q in over)
            + " (z_b1 is correlated with that residual, so the plant adds a cross term; see the calibration section).")


def reading_lines(result, config):
    """DEC-30's Reading, applied mechanically to the primary and the secondaries by the detection rule (interval
    entirely above zero and p below alpha: two-sided sign-flip p for the primary, Holm-adjusted for the secondaries)."""
    tested = {(t['variant'], t['contrast']): t for t in result['tested']}
    primary = next(t for t in result['tested'] if t['role'] == 'primary')
    lines = ['Primary, pre-specified: ' + _tested_text(primary) + '.',
             'Secondary, pre-specified: ' + '; '.join(_tested_text(t) for t in result['tested'] if t['role'] == 'secondary') + '.']
    fired = {v: r['stop_rules']['new']['fired'] for v, r in result['results'].items()}
    flags = {v: r['stop_rules']['fitting_flags'] for v, r in result['results'].items()}
    if any(fired.values()):
        lines.append('No reading: DEC-23\'s stop rule fires (' + '; '.join(f"{v}: {', '.join(c)}" for v, c in fired.items() if c)
                     + '). That signals a fitting problem (optimisation or regularisation) to investigate before interpreting.')
        return lines
    pair = lambda t: f"add_input:{t['full']}-{t['reduced']}"  # noqa: E731
    flagged = [t for t in result['tested'] if pair(t) in flags[t['variant']]]
    if flagged:
        lines.append('Not interpreted (fitting flag: interval entirely below zero): ' + '; '.join(
            f"{CONTRAST_LABELS[t['contrast']]} in {t['variant']}" for t in flagged) + '.')
    if primary in flagged:
        return lines
    f_own, f_partner, f_timing = (tested.get(('F', c)) for c in ('own', 'partner', 'timing'))
    t_partner = tested.get(('T', 'partner'))
    if all(t is not None and t not in flagged for t in (f_own, f_partner, f_timing)):
        if f_own['detectable'] and not f_partner['detectable'] and not f_timing['detectable']:
            lines.append("F: own history > 0 with partner and timing not detectable: RQ1's pattern doesn't depend on the coda types.")
        else:
            lines.append('F: the pattern differs from RQ1\'s (own history detectable: '
                         f"{'yes' if f_own['detectable'] else 'no'}; partner: {'yes' if f_partner['detectable'] else 'no'}; "
                         f"timing: {'yes' if f_timing['detectable'] else 'no'}).")
    t_ok = t_partner is not None and t_partner not in flagged
    if t_ok and t_partner['detectable']:
        lines.append("Partner > 0 in T: the partner's previous codas predict this coda's tempo within its type beyond own history "
                     'and S; consistent with tempo coordination, but onset-available inputs can\'t separate coordination from '
                     'both whales responding to the same context.')
    elif primary['detectable'] and t_ok:
        lines.append('Partner > 0 in F but not in T: the partner predicts which kind of coda (its type-level duration) comes '
                     'next, not its tempo within type.')
    if not primary['detectable']:
        lines.append("Primary null: the partner's previous codas add no detectable information about a coda's duration "
                     'beyond own history and S — not detectable at this data size. ' + bound_text(result['calibration'])
                     + ' A null may also mean too little data: limited data and capacity push contrasts toward zero (PLAN §9).')
    lines.append('Claims are about dyadic exchanges only; nothing here is about grammar or syntax.')
    return lines


def _check_value(value):
    if isinstance(value, dict):
        return ', '.join(f'{k}: {v}' for k, v in value.items())
    if isinstance(value, list):
        return ' / '.join(map(str, value))
    if isinstance(value, float):
        return f'{value:.4g}' if abs(value) < 1e-3 and value != 0 else f'{value:.4f}'
    return str(value)


def report_lines(result, config):
    variants = result['variants']
    fit = result['fit']
    env = fit['environment']
    lines = ['# RQ1 duration lattice (PLAN [P6], DEC-30)', '',
             "RQ1's eight conditions with a type-free target: the coda's log duration y = ln d (d = `offset_s − onset_s`, in "
             f"seconds), for RQ1's {result['comparison_targets']:,} comparison targets, its 14 leave-one-day-out folds with their "
             'validation days, exchanges, S and `[D5]` (all unchanged). Every condition has the always-present inputs: the target\'s '
             'click count n (raw `nClicks`) as ln n, and one-hots for the click counts with at least '
             f"{result['count_one_hot_min_training_targets']} training targets in the fold; variant T adds the target's "
             '`guy_coda_type` one-hot (fold vocabulary, UNK per `[M6]`). A and B: for lags 1–'
             f"{result['history_lags']} of the whale's own and the partner's completed codas, ln d and ln n (standardised on the "
             'non-missing training values, 0 when missing) and a missing indicator; columns constant on the training rows are '
             "dropped. S: logistic v4's S columns. History inputs are type-free in both variants. The floor (`none`) has the "
             'always-present inputs only.', '',
             '**Model.** Ridge regression on y standardised on the fold\'s training targets, in closed form: mean squared error '
             '+ λ_h‖w_A, w_B‖² + λ_S‖w_S‖², a fixed '
             f"{result['always_present_penalty']:g} on the always-present weights and none on the intercept. Penalties from "
             f"logistic v4's grid {result['l2_grid']}, chosen per fold and condition on the validation day by mean held-out log "
             'density (ties: the larger value on each axis); an axis is searched only when its block has columns (S: λ_S; A, B, '
             'A,B: λ_h; A,S, B,S, A,B,S: every pair). σ̂² is the mean squared training residual. A coda\'s score is '
             'log2 N(y; ŷ, σ̂²) in bits per coda; contrasts are paired differences, so the unit of y cancels. Bits per coda of '
             "this continuous target are not comparable with RQ1's bits per coda of a type. Bootstrap: "
             f"{result['bootstrap']['replicates']:,} day-level replicates (seed {result['bootstrap']['seed']}). Added after "
             'the results freeze (2026-10-04) as a final run.', '']

    lines += ['## Data checks', '',
              f"Checked before any model against values computed independently ({result['checks']['source']}): counts exactly, "
              f"other values within {result['checks']['tolerance']} (agreement at 3 decimals). "
              f"{'All reproduce.' if result['checks']['passed'] else 'NOT ALL REPRODUCE.'}", '',
              '| Quantity | Expected | Computed | Reproduced |', '|---|---|---|---|']
    for c in result['checks']['checks']:
        lines.append(f"| {c['name']} | {_check_value(c['expected'])} | {_check_value(c['computed'])} | "
                     f"{'yes' if c['passed'] else 'NO: ' + ', '.join(c['failed_fields'])} |")
    lines += ['', 'Rows whose raw `nClicks` differs from the number of positive ICI1–ICI40 plus one (the run uses `nClicks`):', '',
              '| coda_id | source_row_index | nClicks | positive intervals | guy_coda_type | comparison target |', '|---|---:|---:|---:|---|---|']
    lines += [f"| {r['coda_id']} | {r['source_row_index']} | {r['nClicks']} | {r['positive_intervals']} | {r['guy_coda_type']} | "
              f"{'yes' if r['comparison_target'] else 'no'} |" for r in result['nclicks_mismatch_rows']]
    if result['process_notes']:
        lines += ['', 'Process notes:', ''] + [f'- {note}' for note in result['process_notes']]

    lines += ['', '## Stop rules', '',
              "DEC-23's rule on each variant's 16 checks (the 12 add-one-input contrasts, and the 4 A-conditions against copying "
              "the own last coda's ln d, scored as a Gaussian with the training mean squared difference as variance): exact "
              'one-sided sign-flip test on the day sums (alternative: mean below zero; all 2^14 sign assignments), Holm across '
              "the 16, firing at an adjusted p < 0.05. DEC-16's rule (interval entirely below zero) beside it. An add-one-input "
              'contrast whose interval lies entirely below zero is a fitting flag and is not interpreted.', '',
              '| Variant | DEC-23 rule | DEC-16 rule | Fitting flags |', '|---|---|---|---|']
    for v, r in result['results'].items():
        s = r['stop_rules']
        lines.append(f"| {v} | {', '.join(s['new']['fired']) or 'none fired'} | {', '.join(s['old']) or 'none fired'} | "
                     f"{', '.join(s['fitting_flags']) or 'none'} |")

    lines += ['', '## Primary and secondary contrasts', '',
              'Paired per coda; token-weighted mean (the estimand) and unweighted day mean; day-bootstrap 95% CI; exact two-sided '
              "sign-flip p on the day sums; Holm's correction across the three secondaries. Detectable: interval entirely above "
              'zero and p < 0.05 (Holm-adjusted for the secondaries).', '',
              '| Role | Contrast | Variant | Conditions | Token-weighted | Unweighted day mean | 95% CI | Positive days | Two-sided p | Holm p | Detectable |',
              '|---|---|---|---|---:|---:|---|---:|---:|---:|---|']
    for t in result['tested']:
        lines.append(f"| {t['role']} | {CONTRAST_LABELS[t['contrast']]} | {t['variant']} | {t['full']} − {t['reduced']} | "
                     f"{t['mean_bits_per_coda']:.5f} | {t['unweighted_mean_bits_per_coda']:.5f} | {format_ci(t['ci_bits_per_coda'])} | "
                     f"{t['positive_days']}/{t['days']} | {_p(t['sign_flip_p_two_sided'])} | "
                     f"{_p(t['holm_p']) if 'holm_p' in t else '—'} | {'yes' if t['detectable'] else 'no'} |")
    lines += ['', '## Reading (DEC-30)', ''] + [f'- {line}' for line in result['reading']]

    cal = result['calibration']
    lines += ['', '## Calibration (planted partner effect)', '',
              f"Variant {cal['variant']}, partner contrast only: y is replaced by y + f·σ̂_AS·z_b1 in every split, where z_b1 is the "
              "standardised ln d of the partner's last coda (the A,B,S design's own column, standardised on the training rows) and "
              f"σ̂_AS the fold's training residual SD of condition {cal['sigma_condition'].replace('_', ',')} (median "
              f"{statistics.median(cal['sigma_by_fold'].values()):.4f}); conditions {cal['full'].replace('_', ',')} and "
              f"{cal['reduced'].replace('_', ',')} are refitted with the full penalty search and scored on the planted test day. "
              'The smallest f whose interval lies above zero is the detection bound. The planted contrast includes the real '
              'data\'s own partner contrast; the increment over it is the paired per-coda difference between the planted and the '
              "real-data contrast. DEC-30's nominal size ½·log2(1 + f²) is the increment if z_b1 (unit variance) were independent "
              f"of A, S and the {cal['reduced'].replace('_', ',')} residual; when z_b1 correlates with that residual (the real "
              f"partner signal), planting along it also adds a cross term, which shows as {cal['reduced'].replace('_', ',')}'s "
              'training σ̂ rising by more than √(1 + f²).', '',
              f"| f | DEC-30 nominal size | Planted contrast | Unweighted day mean | 95% CI | Positive days | Two-sided p | Detected | "
              f"Increment over real-data contrast [95% CI] | σ̂ ratio {cal['reduced'].replace('_', ',')} (√(1 + f²)) | "
              f"σ̂ ratio {cal['full'].replace('_', ',')} |",
              '|---:|---:|---:|---:|---|---:|---:|---|---|---:|---:|']
    for p in cal['per_fraction'].values():
        inc = p['increment_over_real']
        lines.append(f"| {p['fraction']} | {p['nominal_maximum_bits_per_coda']:.4f} | {p['mean_bits_per_coda']:.5f} | "
                     f"{p['unweighted_mean_bits_per_coda']:.5f} | {format_ci(p['ci_bits_per_coda'])} | {_days(p)} | "
                     f"{_p(p['sign_flip_p_two_sided'])} | {'yes' if p['detected'] else 'no'} | "
                     f"{inc['mean_bits_per_coda']:.5f} {format_ci(inc['ci_bits_per_coda'])} | "
                     f"{p['sigma_ratio_median'][cal['reduced']]:.4f} ({p['sigma_ratio_if_independent']:.4f}) | "
                     f"{p['sigma_ratio_median'][cal['full']]:.4f} |")
    real = result['results'][cal['variant']]['rq1_contrasts'][cal['contrast']]
    lines += ['', f"Real-data partner contrast in {cal['variant']} (the primary): {real['mean_bits_per_coda']:.5f} bits/coda, "
              f"95% CI {format_ci(real['ci_bits_per_coda'])}. σ̂ ratios are medians over folds of the planted fit's training σ̂ "
              "over the real fit's."]
    lines += ['', bound_text(cal) + (f" Calibration failure (interval below zero) at f = {cal['failed']}." if cal['failed'] else '')]

    lines += ['', '## Lattice', '',
              'Held-out mean log density in bits per coda; the gain over the floor (paired, with day-bootstrap CI); held-out RMS '
              'residual of ln d; median training σ̂ over folds; median chosen penalties with the folds at a grid edge. '
              f"Copy-own-last baseline: {result['copy_own_last']['mean_log_density_bits']:.5f} bits/coda, held-out RMS residual "
              f"{result['copy_own_last']['residuals']['held_out_rms_residual']:.4f}.", '']
    for v, r in result['results'].items():
        lines += [f"### {v} ({variants[v]['label']})", '',
                  '| Condition | Inputs | Bits/coda | Gain over floor | 95% CI | Held-out RMS residual | Median σ̂ | λ_h median (at edge) | λ_S median (at edge) |',
                  '|---|---|---:|---:|---|---:|---:|---|---|']
        for c in result['conditions']:
            m = r['metrics'][c]
            gain = r['versus_floor'].get(c)
            pen = lambda key: 'n/a' if m[key] is None else f"{m[key]['median']:g} ({m[key]['at_grid_edge']}/14)"  # noqa: E731
            lines.append(f"| {c} | {CONDITION_INPUTS[c]} | {m['mean_log_density_bits']:.5f} | "
                         f"{'—' if gain is None else _f(gain['mean_bits_per_coda'])} | {'—' if gain is None else format_ci(gain['ci_bits_per_coda'])} | "
                         f"{m['residuals']['held_out_rms_residual']:.4f} | {m['sigma_median']:.4f} | {pen('chosen_l2')} | {pen('chosen_l2_s')} |")
        lines.append('')

    lines += ['## All twelve add-one-input contrasts', '']
    for v, r in result['results'].items():
        lines += [f'### {v}', '', '| Pair | Token-weighted | Unweighted day mean | 95% CI | Positive days | Fitting flag |',
                  '|---|---:|---:|---|---:|---|']
        for name, s in r['add_one_input'].items():
            lines.append(f"| {name.replace('-', ' − ')} | {s['mean_bits_per_coda']:.5f} | {s['unweighted_mean_bits_per_coda']:.5f} | "
                         f"{format_ci(s['ci_bits_per_coda'])} | {_days(s)} | {'yes' if s['ci_bits_per_coda'][1] < 0 else 'no'} |")
        lines.append('')

    lines += ['## By turn category and by repeat (descriptive)', '',
              "RQ1's three contrasts within groups of targets: token-weighted mean and positive days; no CIs or tests. Repeat: the "
              "target's stored type equals the whale's own last coda's.", '',
              '| Variant | Contrast | Group | Targets | Token-weighted | Positive days |', '|---|---|---|---:|---:|---:|']
    for v, r in result['results'].items():
        for name in CONTRASTS:
            for group, s in r['by_turn_category'][name].items():
                lines.append(f"| {v} | {CONTRAST_LABELS[name]} | {group} | {s['codas']} | {s['mean_bits_per_coda']:.5f} | {s['positive_days']}/{s['days']} |")
            for group, s in r['by_repeat'][name].items():
                label = 'repeats own last type' if group == 'True' else 'switches type'
                lines.append(f"| {v} | {CONTRAST_LABELS[name]} | {label} | {s['codas']} | {s['mean_bits_per_coda']:.5f} | {s['positive_days']}/{s['days']} |")

    lines += ['', '## Weights on the last coda\'s ln d (fold summaries)', '',
              "Fitted change in predicted ln d per unit of the own (A) or partner (B) last coda's ln d, on the original scale "
              '(1 = full copy, given the other inputs); min, median and max over the 14 folds.', '',
              '| Variant | Condition | Own lag 1 (min / median / max) | Partner lag 1 (min / median / max) |', '|---|---|---|---|']
    for v, r in result['results'].items():
        for c, s in r['lag1_slopes'].items():
            text = lambda b: '—' if b not in s else f"{s[b]['min']:.3f} / {s[b]['median']:.3f} / {s[b]['max']:.3f}"  # noqa: E731
            lines.append(f'| {v} | {c} | {text("A")} | {text("B")} |')

    lines += ['', '## Residual diagnostics', '',
              "Held-out residuals standardised by each fold's σ̂, pooled over the test days. A Gaussian has skewness 0, excess "
              'kurtosis 0, 0.27% beyond 3σ and mean squared standardised residual 1.', '',
              '| Model | Condition | Skewness | Excess kurtosis | Beyond 3σ̂ | Mean squared standardised residual |', '|---|---|---:|---:|---:|---:|']
    for v, r in result['results'].items():
        for c in result['conditions']:
            d = r['metrics'][c]['residuals']
            lines.append(f"| {v} | {c} | {d['skewness']:.3f} | {d['excess_kurtosis']:.3f} | {d['beyond_3_sigma']} "
                         f"({100 * d['share_beyond_3_sigma']:.2f}%) | {d['mean_squared_standardised_residual']:.3f} |")
    d = result['copy_own_last']['residuals']
    lines.append(f"| copy own last | — | {d['skewness']:.3f} | {d['excess_kurtosis']:.3f} | {d['beyond_3_sigma']} "
                 f"({100 * d['share_beyond_3_sigma']:.2f}%) | {d['mean_squared_standardised_residual']:.3f} |")

    lines += ['', '## v5 line (descriptive)', '',
              "DEC-23's absent candidates applied to these fits, no new fits: per fold and condition, the sub-condition fit with "
              'the highest validation-day mean log density (exact ties: fewer blocks); its test-day scores become the '
              "condition's. Not tested; no Holm.", '']
    for v, r in result['results'].items():
        lines += [f'### {v}', '', '| Condition | v5 bits/coda | Sub-conditions chosen (folds) |', '|---|---:|---|']
        for c in result['conditions']:
            counts = ', '.join(f'{k}: {n}' for k, n in r['v5']['selection_counts'][c].items())
            lines.append(f"| {c} | {r['v5']['mean_log_density_bits'][c]:.5f} | {counts} |")
        lines += ['', '| Contrast | Token-weighted | Unweighted day mean | 95% CI | Positive days | Two-sided p |', '|---|---:|---:|---|---:|---:|']
        for name, s in r['v5']['contrasts'].items():
            lines.append(f"| {CONTRAST_LABELS[name]} ({s['full']} − {s['reduced']}) | {s['mean_bits_per_coda']:.5f} | "
                         f"{s['unweighted_mean_bits_per_coda']:.5f} | {format_ci(s['ci_bits_per_coda'])} | {_days(s)} | {_p(s['sign_flip_p_two_sided'])} |")
        lines.append('')

    lines += ['## DEC-23 checks per variant', '',
              '| Variant | Check | Token-weighted | 95% CI | Positive days | One-sided p | Holm p | DEC-23 fires | DEC-16 fires |',
              '|---|---|---:|---|---:|---:|---:|---|---|']
    for v, r in result['results'].items():
        for row in r['stop_rules']['new']['checks']:
            lines.append(f"| {v} | {row['check']} | {row['mean_bits_per_coda']:.5f} | {format_ci(row['ci_bits_per_coda'])} | "
                         f"{row['positive_days']}/{row['days']} | {_p(row['sign_flip_p'])} | {_p(row['holm_p'])} | "
                         f"{'yes' if row['fires'] else 'no'} | {'yes' if row['old_rule_fires'] else 'no'} |")

    git = fit['git']
    lines += ['', '## Limits', '',
              "- The partner's coda that overlaps the target isn't complete at the target's onset and isn't an input, so duration "
              'matching within overlaps is outside this design.',
              '- Gaussian residuals: heavy tails affect every condition but can inflate differences; diagnostics above.',
              "- Bits per coda of a continuous target are not comparable with RQ1's bits per coda of a type.",
              '- Ornaments (an extra final click) aren\'t flagged: conditioning on click count groups an ornamented coda with codas '
              'of its click count, and in T many ornamented codas have their own type.',
              '- Added after the results freeze (2026-10-04) as a final run.',
              '- Not done: heavy-tailed or nonlinear models; the residual LSTM; inter-click-interval targets; other inventories in T.',
              '', '## Run', '',
              f"{fit['fits']:,} penalty candidates in the fit stage ({', '.join(f'{v} {s:.1f} s' for v, s in fit['fit_seconds'].items())}; "
              f"{fit['total_seconds']:.1f} s with data preparation and the baseline), {fit['calibration_fits']:,} in the calibration "
              f"({fit['calibration_seconds']:.1f} s); Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, one BLAS "
              f"thread; code at commit {git['commit'] or 'n/a'}. Outputs in `{config['output_directory']}/` (`checks/`, `fit/`, "
              "`calibration/`); `summary.json` records SHA-256 of every input to this report, as listed in the config."]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config/rq1_duration_lattice.json')
    parser.add_argument('--stage', choices=('all',) + STAGES, default='all')
    args = parser.parse_args()
    if args.stage in ('all', 'checks'):
        run_checks(args.config)
    if args.stage in ('all', 'fit'):
        run_fit(args.config)
    if args.stage in ('all', 'calibrate'):
        run_calibration(args.config)
    if args.stage in ('all', 'report'):
        print('\n'.join(run_report(args.config)['reading']))


if __name__ == '__main__':
    main()
