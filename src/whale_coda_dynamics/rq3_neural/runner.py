"""Stages of the pooled RQ3 LSTM experiment (PLAN [R3-8], DEC-27).

    PYTHONPATH=src python -m whale_coda_dynamics.rq3_neural.runner --config config/rq3_neural.json --stage <stage>

Stages, in order (each reads only the outputs of the stages before it):
- ``pilot``: gate 0 (coda counts per clan and fold against rq3_folds.csv); one own-clan model and
  one pooled run over the full grid (no held-out codas are scored); device and throughput
  measurement; projected compute time and, if it is over budget, the reduced arm 5.
- ``arm1``: own-clan LSTMs for every clan and fold, gate 1 against the own-clan n-gram (with the
  one rerun [R3-8] allows), and every fold model scored on the other eight clans.
- ``arm23``: pooled tagged (arm 2) and untagged (arm 3) models for every alpha and run; primary alpha.
- ``capacity``: the [R3-8] capacity check (arm 2 rerun with double the hidden size where pooled
  tagged is reliably below own-clan).
- ``arm4``: leave one clan out, at arm 3's primary alpha.
- ``arm5``: data efficiency, fine-tuned from arm 4 against from scratch.
- ``report``: contrasts, tables and reports/rq3_neural.md (``report.py``).

Selection fits use one seed; the chosen setting is refit with every refit seed for the selected fit's
best epoch, and a coda's score is its mean log-likelihood over the refit seeds. Every task passes the
leakage checks in ``data.py`` before it is submitted; each stage refits one model a second time and
requires identical scores.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from whale_coda_dynamics.preprocessing.io_utils import read_json, write_text
from whale_coda_dynamics.rq3.experiments.data import read_csv, write_csv
from whale_coda_dynamics.rq3.experiments.uncertainty import pair_seed

from .contrasts import ci_below_zero, paired_contrast
from .data import (
    check_disjoint,
    check_excludes_clan,
    check_n_symbols,
    check_train_val_disjoint,
    gate0_rows,
    loo_plan,
    own_clan_plan,
    pooled_plan,
    pooled_runs,
    Context,
    load_vlad_token_counts,
)
from .model import FitPool, sampling_probabilities, warm_pool

CODE_ROOT = Path(__file__).resolve().parents[3]
STAGES = ('pilot', 'arm1', 'arm23', 'capacity', 'arm4', 'arm5', 'report')
SCORE_COLUMNS = ['run_id', 'model', 'tokeniser', 'train_clans', 'train_folds', 'target_clan', 'rq3_coda_id',
                 'target_fold', 'n_symbols', 'll_bits', 'll_eos_bits']
SELECTION_COLUMNS = ['arm', 'model', 'run', 'candidate', 'hidden', 'dropout', 'learning_rate', 'patience', 'seed',
                     'alpha', 'tagged', 'n_train', 'n_inner_validation', 'best_epoch', 'epochs_trained',
                     'early_stopped', 'epoch0_bits_per_symbol', 'inner_validation_bits_per_symbol', 'selected',
                     'seconds']
INNER_COLUMNS = ['arm', 'model', 'run', 'candidate', 'clan', 'll_bits', 'n_symbols', 'bits_per_symbol']
REFIT_COLUMNS = ['arm', 'model', 'run', 'candidate', 'seed', 'epochs', 'n_train', 'init_sha256', 'state_sha256',
                 'seconds']
GROUP_SUM_COLUMNS = ['run_id', 'model', 'tokeniser', 'train_clans', 'train_folds', 'target_clan', 'target_fold',
                     'share', 'draw', 'version', 'learning_rate', 'best_epoch', 'group_id', 'n_codas', 'n_symbols',
                     'll_bits_sum', 'll_eos_bits_sum']
GATE1_COLUMNS = ['clan', 'n_codas', 'n_groups', 'lstm_bits_per_symbol', 'ngram_bits_per_symbol',
                 'difference_bits_per_symbol', 'ci95_lower_bits_per_symbol', 'ci95_upper_bits_per_symbol', 'passed']
VALIDATION_COST = 0.5  # projection only: one validation coda costs about half a training draw


# ---------------------------------------------------------------- bookkeeping

def git_state():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=CODE_ROOT, capture_output=True, text=True,
                                check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty_paths=None)
    return dict(commit=commit, dirty_paths=[line for line in status if not line[3:].startswith('.claude/')])


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_text(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def output_dir(ctx, name):
    return ctx.root / ctx.config['output_directory'] / name


class Stage:
    """One stage's output directory, inputs and summary.json (config, input SHA-256, commit, timing)."""

    def __init__(self, ctx, name, git):
        self.ctx, self.name, self.git = ctx, name, git
        self.dir = output_dir(ctx, name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.inputs = [ctx.config_path, ctx.path('codas_input'), ctx.path('folds_input'),
                       ctx.path('ngram_self_scores_input')]
        self.fit_seconds, self.fit_count = 0.0, 0

    def add_inputs(self, *paths):
        self.inputs.extend(Path(p) for p in paths)

    def record(self, results):
        self.fit_seconds += sum(r['seconds'] for r in results)
        self.fit_count += len(results)

    def csv(self, name, columns, rows):
        path = self.dir / name
        write_csv(path, columns, rows)
        return path

    def finish(self, **extra):
        root = self.ctx.root
        summary = dict(stage=self.name, config=self.ctx.config,
                       input_sha256={Path(p).resolve().relative_to(root).as_posix(): sha256_file(p)
                                     for p in dict.fromkeys(Path(p).resolve() for p in self.inputs)},
                       commit=self.git['commit'], dirty_paths=self.git['dirty_paths'],
                       python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                       started=datetime.fromtimestamp(self.started, timezone.utc).isoformat(timespec='seconds'),
                       finished=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                       wall_seconds=round(time.time() - self.started, 1), fits=self.fit_count,
                       fit_seconds=round(self.fit_seconds, 1), **extra)
        write_json(self.dir / 'summary.json', summary)
        return summary


def check_tree(ctx, git):
    if ctx.config['smoke_test'] and ctx.root.resolve() == CODE_ROOT:
        raise ValueError('A smoke-test config must not write into the repository')
    if ctx.config['require_clean_tree'] and (git['commit'] is None or git['dirty_paths']):
        raise ValueError(f'Committed results come from a clean tree; uncommitted: {git["dirty_paths"]}')


def execution(ctx):
    """Device and worker count: from the config, or the pilot's decision when device is 'auto'."""
    if ctx.config['device'] != 'auto':
        return ctx.config['device'], ctx.config['workers']
    decision = read_json(output_dir(ctx, 'pilot') / 'summary.json')['decision']
    if decision.get('stop'):
        raise ValueError('The pilot projected more than the stop budget; nothing else runs')
    return decision['device'], decision['workers']


def set_cuda_environment(ctx, device):
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ctx.config['pilot']['cublas_workspace_config'])


# ---------------------------------------------------------------- candidates and tasks

def candidate_grid(config, extended=False):
    model = config['model']
    if extended:
        rerun = config['gate_rerun']
        grid = [dict(hidden=h, dropout=d, learning_rate=lr, patience=rerun['patience'])
                for h in rerun['hidden_grid'] for d in model['dropout_grid'] for lr in rerun['learning_rates']]
    else:
        grid = [dict(hidden=h, dropout=d, learning_rate=model['learning_rate'], patience=model['patience'])
                for h in model['hidden_grid'] for d in model['dropout_grid']]
    for index, candidate in enumerate(grid):
        candidate['index'] = index
        candidate['name'] = name_of(candidate)
    return grid


def name_of(candidate):
    return f"h{candidate['hidden']}_d{candidate['dropout']}_lr{candidate['learning_rate']}_p{candidate['patience']}"


def make_task(ctx, key, *, train, val, guard, tagged, alpha, candidate, seed, epochs, device, score=None,
              inherited=(), init_checkpoint=None, init_checkpoint_sha256=None, save_checkpoint=None,
              max_epochs=None, record_draws=True):
    """One fit, after the leakage checks: training (and inherited training) and validation codas are
    disjoint from the guarded codas (the plan's held-out codas) and from every coda the fit scores."""
    label = ':'.join(map(str, key))
    check_disjoint(label, ctx, guard, train, val, *inherited)
    if score is not None:
        check_disjoint(label, ctx, score, train, val, *inherited)
    if val is not None:
        check_train_val_disjoint(label, ctx, train, val)
    if len(train) == 0:
        raise ValueError(f'{label}: no training codas')
    model = ctx.config['model']
    expected_epochs = epochs if epochs is not None else 30
    return dict(key=key, train_idx=np.asarray(train, dtype=np.int64),
                val_idx=None if val is None else np.asarray(val, dtype=np.int64),
                score_idx=None if score is None else np.asarray(score, dtype=np.int64),
                tagged=bool(tagged), alpha=float(alpha), hidden=candidate['hidden'], dropout=candidate['dropout'],
                learning_rate=candidate['learning_rate'], patience=candidate['patience'],
                max_epochs=max_epochs if max_epochs is not None else model['max_epochs'],
                batch_size=model['batch_size'], embedding_dim=model['embedding_dim'],
                eval_batch_size=model['eval_batch_size'], seed=int(seed), epochs=epochs, device=device,
                init_checkpoint=None if init_checkpoint is None else str(init_checkpoint),
                init_checkpoint_sha256=init_checkpoint_sha256,
                save_checkpoint=None if save_checkpoint is None else str(save_checkpoint),
                record_draws=record_draws,
                cost=len(train) * expected_epochs * (1 + candidate['hidden'] / 64))


def choose(candidates, results):
    """Best inner-validation bits per symbol; ties go to the smaller model (hidden, then dropout)."""
    best = max(range(len(candidates)), key=lambda i: (results[i]['inner_validation']['bits_per_symbol'],
                                                      -candidates[i]['hidden'], -candidates[i]['dropout'], -i))
    return best


def require_trained(label, result):
    if result['best_epoch'] < 1:
        raise ValueError(f'{label}: the best inner-validation epoch is 0 (the initial weights), a fitting failure')


def selection_row(arm, model, run, candidate, task, result, selected):
    inner = result['inner_validation']
    return dict(arm=arm, model=model, run=run, candidate=candidate['name'], hidden=candidate['hidden'],
                dropout=candidate['dropout'], learning_rate=candidate['learning_rate'],
                patience=candidate['patience'], seed=task['seed'], alpha=task['alpha'], tagged=task['tagged'],
                n_train=result['n_train'], n_inner_validation=len(task['val_idx']), best_epoch=result['best_epoch'],
                epochs_trained=result['epochs_trained'], early_stopped=result['early_stopped'],
                epoch0_bits_per_symbol=result['curve'][0], inner_validation_bits_per_symbol=inner['bits_per_symbol'],
                selected=selected, seconds=round(result['seconds'], 3))


def inner_rows(ctx, arm, model, run, candidate, result):
    rows = []
    for code, (ll, n) in sorted(result['inner_validation']['by_clan'].items()):
        rows.append(dict(arm=arm, model=model, run=run, candidate=candidate['name'], clan=ctx.clans[code],
                         ll_bits=ll, n_symbols=n, bits_per_symbol=ll / n))
    return rows


def refit_row(arm, model, run, candidate, task, result):
    return dict(arm=arm, model=model, run=run, candidate=candidate['name'], seed=task['seed'], epochs=task['epochs'],
                n_train=result['n_train'], init_sha256=result['init_sha256'], state_sha256=result['state_sha256'],
                seconds=round(result['seconds'], 3))


def check_scored(ctx, label, task, result):
    """The model scored every requested coda with exactly Vlad's token_count symbols."""
    check_n_symbols(label, task['score_idx'], result['scores']['n_symbols'], ctx)


def check_refit_reproducible(pool, task, result, label):
    """Refitting one model with the same seed gives identical scores (and weights)."""
    again = pool.run([dict(task, save_checkpoint=None)])[0]
    same = (again['state_sha256'] == result['state_sha256']
            and np.array_equal(again['scores']['ll_bits'], result['scores']['ll_bits'])
            and np.array_equal(again['scores']['ll_eos_bits'], result['scores']['ll_eos_bits']))
    if not same:
        raise ValueError(f'{label}: refitting with the same seed gave different scores')
    return dict(model=':'.join(map(str, task['key'])), seed=task['seed'], identical=True,
                state_sha256=result['state_sha256'])


def check_same_draws(label, a_task, a, b_task, b, fields_may_differ):
    """Two fits that may differ only in ``fields_may_differ``: same training codas, sampling, seed and
    settings, the same initial weights unless the initialisation may differ, and the same draws over
    the epochs both ran."""
    ignore = {'key', 'cost', 'save_checkpoint', *fields_may_differ}
    for field in set(a_task) | set(b_task):
        if field in ignore:
            continue
        x, y = a_task.get(field), b_task.get(field)
        equal = np.array_equal(x, y) if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) else x == y
        if not equal:
            raise ValueError(f'{label}: the paired fits differ in {field}')
    if 'init_checkpoint' not in fields_may_differ and a['init_sha256'] != b['init_sha256']:
        raise ValueError(f'{label}: the paired fits start from different weights')
    if a['sampling_q'] != b['sampling_q'] or a['n_train'] != b['n_train']:
        raise ValueError(f'{label}: the paired fits sample differently')
    common = min(len(a['draw_hashes']), len(b['draw_hashes']))
    if common < 1 or a['draw_hashes'][:common] != b['draw_hashes'][:common]:
        raise ValueError(f'{label}: the paired fits saw different training draws')
    return common


def score_rows(ctx, index, ll, eos, run_id, model, train_clans, train_folds):
    rows = []
    for i, coda in enumerate(index):
        rows.append(dict(run_id=run_id, model=model, tokeniser=ctx.config['tokenizer_id'], train_clans=train_clans,
                         train_folds=train_folds, target_clan=ctx.clan_of[coda], rq3_coda_id=ctx.ids[coda],
                         target_fold=int(ctx.fold_of[coda]), n_symbols=int(ctx.n_symbols[coda]),
                         ll_bits=float(ll[i]), ll_eos_bits=float(eos[i])))
    return rows


def read_scores(ctx, path):
    """Score rows back as {model: {coda_id: (ll_bits, n_symbols)}}, checking n_symbols against Vlad's."""
    rows = read_csv(path, SCORE_COLUMNS)
    out = {}
    for row in rows:
        n = int(row['n_symbols'])
        if n != ctx.vlad_token_count[row['rq3_coda_id']]:
            raise ValueError(f"{path}: n_symbols of {row['rq3_coda_id']} differs from Vlad's token_count")
        scores = out.setdefault(row['model'], {})
        if row['rq3_coda_id'] in scores:
            raise ValueError(f"{path}: {row['model']} scores {row['rq3_coda_id']} twice")
        scores[row['rq3_coda_id']] = (float(row['ll_bits']), n)
    return out


def transfer_model_name(source):
    return f'lstm_own_transfer_from_{source}'


def fold_label(folds):
    return ' '.join(str(f) for f in folds)


def mean_over_seeds(results):
    return (np.mean([r['scores']['ll_bits'] for r in results], axis=0),
            np.mean([r['scores']['ll_eos_bits'] for r in results], axis=0))


# ---------------------------------------------------------------- pilot

def stage_pilot(ctx, git):
    config, pilot = ctx.config, ctx.config['pilot']
    stage = Stage(ctx, 'pilot', git)
    gate0 = gate0_rows(ctx)
    stage.csv('gate0.csv', ['clan', 'fold', 'codas_file', 'fold_file', 'encoded', 'match'], gate0)
    print(f'Gate 0 passed: {len(gate0)} clan/fold counts match rq3_folds.csv', flush=True)
    grid = candidate_grid(config)
    model = config['model']
    own = own_clan_plan(ctx, pilot['own_clan'], pilot['own_fold'])
    pooled = pooled_plan(ctx, pilot['pooled_run'])
    alpha, tagged = pilot['pooled_alpha'], pilot['pooled_tagged']

    def pilot_tasks(device, max_epochs=None, patience=None, seeds=None):
        tasks = []
        for name, plan, a, t in (('own', own, 1.0, False), ('pooled', pooled, alpha, tagged)):
            for candidate in grid:
                c = dict(candidate) if patience is None else dict(candidate, patience=patience)
                tasks.append(make_task(ctx, ('pilot', name, candidate['name']), train=plan['select_train'],
                                       val=plan['val'], guard=plan['held'], tagged=t, alpha=a, candidate=c,
                                       seed=model['selection_seed'], epochs=None, device=device,
                                       max_epochs=max_epochs))
        return tasks

    # The pilot fits themselves (CPU): full grid, then the refits, without scoring any held-out coda.
    cpu_max = max(pilot['cpu_workers'])
    fit_rows = []
    with FitPool(ctx.encoded, cpu_max, 'cpu') as pool:
        warm_pool(pool)
        tasks = pilot_tasks('cpu')
        results = pool.run(tasks)
        stage.record(results)
        estimates = {}
        for name, plan, a, t in (('own', own, 1.0, False), ('pooled', pooled, alpha, tagged)):
            idx = [i for i, task in enumerate(tasks) if task['key'][1] == name]
            best = choose([grid[i % len(grid)] for i in idx], [results[i] for i in idx])
            chosen = results[idx[best]]
            require_trained(f'pilot {name}', chosen)
            refit_tasks = [make_task(ctx, ('pilot', name, 'refit', s), train=plan['refit_train'], val=None,
                                     guard=plan['held'], tagged=t, alpha=a, candidate=grid[best], seed=s,
                                     epochs=chosen['best_epoch'], device='cpu') for s in model['refit_seeds']]
            refits = pool.run(refit_tasks)
            stage.record(refits)
            for i in idx:
                fit_rows.append(dict(model=name, candidate=grid[i % len(grid)]['name'], kind='selection',
                                     n_train=results[i]['n_train'], epochs=results[i]['epochs_trained'],
                                     best_epoch=results[i]['best_epoch'], seconds=round(results[i]['seconds'], 3),
                                     draws=results[i]['draws']))
            for task, r in zip(refit_tasks, refits):
                fit_rows.append(dict(model=name, candidate=grid[best]['name'], kind=f"refit seed {task['seed']}",
                                     n_train=r['n_train'], epochs=task['epochs'], best_epoch=task['epochs'],
                                     seconds=round(r['seconds'], 3), draws=r['draws']))
            estimates[name] = dict(selection_epochs=float(np.mean([results[i]['epochs_trained'] for i in idx])),
                                   best_epochs=float(np.mean([results[i]['best_epoch'] for i in idx])))
    stage.csv('fits.csv', ['model', 'candidate', 'kind', 'n_train', 'epochs', 'best_epoch', 'seconds', 'draws'],
              fit_rows)

    # Determinism and throughput: fixed-length timing fits of the pooled run on every option.
    timing_epochs = pilot['timing_epochs']
    probe = pilot_tasks('cpu', max_epochs=timing_epochs, patience=timing_epochs + 1)[len(grid)]
    options, timing_rows, determinism = [], [], []

    def measure(device, workers):
        rates = {}
        with FitPool(ctx.encoded, workers, device) as pool:
            warm_pool(pool)
            for hidden in model['hidden_grid']:
                c = dict(grid[0], hidden=hidden, patience=timing_epochs + 1)
                tasks = [make_task(ctx, ('timing', device, workers, hidden, w), train=pooled['select_train'],
                                   val=pooled['val'], guard=pooled['held'], tagged=tagged, alpha=alpha, candidate=c,
                                   seed=w, epochs=None, device=device, max_epochs=timing_epochs)
                         for w in range(workers)]
                started = time.time()
                results = pool.run(tasks, order_by_cost=False)
                wall = time.time() - started
                rate = sum(r['draws'] / r['seconds'] for r in results)
                rates[hidden] = rate
                timing_rows.append(dict(device=device, workers=workers, hidden=hidden, fits=len(results),
                                        epochs=timing_epochs, wall_seconds=round(wall, 2),
                                        draws_per_second=round(rate, 1)))
                print(f'  {device} x{workers} hidden {hidden}: {rate:,.0f} draws/s', flush=True)
        return rates

    def deterministic(device):
        outcomes = []
        for _ in range(2):  # two separate processes
            with FitPool(ctx.encoded, 1, device) as pool:
                outcomes.append(pool.run([dict(probe, device=device)])[0])
        # Identical weights and identical inner-validation log-likelihoods after every epoch.
        same = (outcomes[0]['state_sha256'] == outcomes[1]['state_sha256']
                and outcomes[0]['curve'] == outcomes[1]['curve']
                and outcomes[0]['inner_validation']['by_clan'] == outcomes[1]['inner_validation']['by_clan'])
        determinism.append(dict(device=device, seed=probe['seed'], identical=bool(same),
                                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')))
        return same

    if not deterministic('cpu'):
        raise ValueError('CPU fits with the same seed differ between processes')
    for workers in pilot['cpu_workers']:
        options.append(dict(device='cpu', workers=workers, rates=measure('cpu', workers)))
    if pilot['gpu_workers'] and torch.cuda.is_available():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', pilot['cublas_workspace_config'])
        if deterministic('cuda'):
            for workers in pilot['gpu_workers']:
                options.append(dict(device='cuda', workers=workers, rates=measure('cuda', workers)))
    else:
        determinism.append(dict(device='cuda', seed=probe['seed'], identical=None,
                                cublas_workspace_config='not available' if not torch.cuda.is_available() else 'not tried'))
    stage.csv('timing.csv', ['device', 'workers', 'hidden', 'fits', 'epochs', 'wall_seconds', 'draws_per_second'],
              timing_rows)
    stage.csv('determinism.csv', ['device', 'seed', 'identical', 'cublas_workspace_config'], determinism)

    arm5_full = dict(shares=config['arm5']['shares'], draws=config['arm5']['draws'], reduced=False)
    for option in options:
        option['projection'] = projection(ctx, option['rates'], estimates, arm5_full)
    fastest = min(options, key=lambda o: o['projection']['total_hours'])
    cpu_best = min((o for o in options if o['device'] == 'cpu'), key=lambda o: o['projection']['total_hours'])
    chosen = cpu_best if cpu_best['projection']['total_hours'] <= fastest['projection']['total_hours'] * (
        1 + pilot['tie_tolerance']) else fastest
    decision = dict(device=chosen['device'], workers=chosen['workers'], arm5=arm5_full, stop=False,
                    projection=chosen['projection'])
    print(f"Chosen: {chosen['device']} x{chosen['workers']}; projected total "
          f"{chosen['projection']['total_hours']:.2f} h", flush=True)
    if chosen['projection']['total_hours'] > pilot['budget_hours']:
        reduced = dict(shares=config['arm5']['reduced_shares'], draws=config['arm5']['reduced_draws'], reduced=True)
        decision['arm5'] = reduced
        decision['projection'] = projection(ctx, chosen['rates'], estimates, reduced)
        print(f"Over {pilot['budget_hours']} h: arm 5 reduced to shares {reduced['shares']} with "
              f"{reduced['draws']} draw(s); new projection {decision['projection']['total_hours']:.2f} h", flush=True)
        if decision['projection']['total_hours'] > pilot['stop_hours']:
            decision['stop'] = True
    summary = stage.finish(gate0_passed=True, estimates=estimates, options=options, decision=decision,
                           determinism=determinism)
    if decision['stop']:
        raise SystemExit(f"Projected {decision['projection']['total_hours']:.2f} h exceeds the stop budget of "
                         f"{pilot['stop_hours']} h even with the reduced arm 5")
    return summary


def projection(ctx, rates, estimates, arm5):
    """Projected wall hours per arm from aggregate draws per second (all workers busy).

    Fits cost (training draws + VALIDATION_COST x validation codas per epoch) / rate; selection fits
    run the pilot's mean number of epochs, refits its mean best epoch. Conditional work (the gate
    rerun and the capacity check) is not included."""
    config, model = ctx.config, ctx.config['model']
    hidden = model['hidden_grid']
    seeds = len(model['refit_seeds'])
    mean_rate = len(hidden) / sum(1 / rates[h] for h in hidden)
    own_e, pool_e = estimates['own'], estimates['pooled']

    def selection(n_train, n_val, e):
        return sum((n_train + VALIDATION_COST * n_val) * e['selection_epochs'] / rates[h] for h in hidden)

    def refits(n_train, n_scored, e):
        return seeds * (n_train * e['best_epochs'] + VALIDATION_COST * n_scored) / mean_rate

    seconds = dict(arm1=0.0, arm23=0.0, arm4=0.0, arm5=0.0)
    total = len(ctx.ids)
    for clan in ctx.clans:
        for k in range(ctx.n_folds[clan]):
            plan = own_clan_plan(ctx, clan, k)
            seconds['arm1'] += selection(len(plan['select_train']), len(plan['val']), own_e)
            seconds['arm1'] += refits(len(plan['refit_train']), total - len(plan['refit_train']), own_e)
            for f in arm5['shares']:
                m = max(1, round(f * len(plan['select_train'])))
                per_fit = (m + VALIDATION_COST * len(plan['val'])) * own_e['selection_epochs'] / mean_rate
                seconds['arm5'] += arm5['draws'] * 2 * len(config['arm5']['learning_rates']) * per_fit
    for run in pooled_runs(ctx):
        plan = pooled_plan(ctx, run)
        per_arm = selection(len(plan['select_train']), len(plan['val']), pool_e) + refits(
            len(plan['refit_train']), len(plan['held']), pool_e)
        seconds['arm23'] += 2 * len(config['pooled_alphas']) * per_arm
    for clan in ctx.clans:
        plan = loo_plan(ctx, clan)
        seconds['arm4'] += selection(len(plan['select_train']), len(plan['val']), pool_e) + refits(
            len(plan['refit_train']), len(plan['held']), pool_e)
    hours = {arm: s / 3600 for arm, s in seconds.items()}
    return dict(hours=hours, total_hours=sum(hours.values()), arm5=arm5)


# ---------------------------------------------------------------- arm 1

def arm1_source(ctx):
    """The arm 1 run the later arms use: the first run if gate 1 passed, else the rerun."""
    first = read_json(output_dir(ctx, 'arm1') / 'summary.json')
    if first['gate_passed']:
        return 'arm1'
    rerun_path = output_dir(ctx, 'arm1_rerun') / 'summary.json'
    if rerun_path.is_file() and (read_json(rerun_path)['gate_passed'] or ctx.config['smoke_test']):
        return 'arm1_rerun'
    if ctx.config['smoke_test']:
        return 'arm1'
    raise ValueError('Gate 1 failed for arm 1 and its rerun: fix the fit before running any contrast')


def run_arm1(ctx, stage, pool, device, extended):
    model = ctx.config['model']
    grid = candidate_grid(ctx.config, extended)
    plans = {(clan, k): own_clan_plan(ctx, clan, k) for clan in ctx.clans for k in range(ctx.n_folds[clan])}
    tasks, owners = [], []
    for (clan, k), plan in plans.items():
        for candidate in grid:
            tasks.append(make_task(ctx, ('arm1', clan, k, candidate['name']), train=plan['select_train'],
                                   val=plan['val'], guard=plan['held'], tagged=False, alpha=1.0,
                                   candidate=candidate, seed=model['selection_seed'], epochs=None, device=device))
            owners.append((clan, k, candidate))
    print(f'{stage.name}: {len(tasks)} selection fits', flush=True)
    results = pool.run(tasks)
    stage.record(results)
    selection, chosen = [], {}
    for (clan, k), plan in plans.items():
        idx = [i for i, o in enumerate(owners) if o[:2] == (clan, k)]
        best = choose([owners[i][2] for i in idx], [results[i] for i in idx])
        chosen[clan, k] = (owners[idx[best]][2], results[idx[best]])
        require_trained(f'arm1 {clan} fold {k}', results[idx[best]])
        for j, i in enumerate(idx):
            selection.append(selection_row('arm1', 'lstm_own', f'{clan}:fold{k}', owners[i][2], tasks[i], results[i],
                                           j == best))
    refit_tasks, refit_owners = [], []
    for (clan, k), plan in plans.items():
        candidate, result = chosen[clan, k]
        other = np.concatenate([ctx.clan_indices(c) for c in ctx.clans if c != clan])
        for seed in model['refit_seeds']:
            refit_tasks.append(make_task(ctx, ('arm1', clan, k, candidate['name'], 'refit'), train=plan['refit_train'],
                                         val=None, guard=plan['held'], score=np.concatenate([plan['held'], other]),
                                         tagged=False, alpha=1.0, candidate=candidate, seed=seed,
                                         epochs=result['best_epoch'], device=device))
            refit_owners.append((clan, k, candidate, len(plan['held']), other))
    print(f'{stage.name}: {len(refit_tasks)} refits', flush=True)
    refits = pool.run(refit_tasks)
    stage.record(refits)
    for task, result in zip(refit_tasks, refits):
        check_scored(ctx, ':'.join(map(str, task['key'])), task, result)
    reproducible = check_refit_reproducible(pool, refit_tasks[0], refits[0], 'arm1')
    own_rows, matrix_rows, refit_rows, own = [], [], [], {}
    fold_models = {}
    for (clan, k), plan in plans.items():
        group = [(t, r, o) for t, r, o in zip(refit_tasks, refits, refit_owners) if o[:2] == (clan, k)]
        candidate, n_held = group[0][2][2], group[0][2][3]
        held_results = [dict(scores={key: r['scores'][key][:n_held] for key in r['scores']}) for _, r, _ in group]
        other_results = [dict(scores={key: r['scores'][key][n_held:] for key in r['scores']}) for _, r, _ in group]
        ll, eos = mean_over_seeds(held_results)
        own_rows += score_rows(ctx, plan['held'], ll, eos, f'arm1:{clan}:fold{k}', 'lstm_own', clan,
                               fold_label(plan['refit_folds'][clan]))
        for i, coda in enumerate(plan['held']):
            own[ctx.ids[coda]] = (float(ll[i]), int(ctx.n_symbols[coda]))
        fold_models.setdefault(clan, []).append((group[0][2][4], mean_over_seeds(other_results)))
        for task, result, _ in group:
            refit_rows.append(refit_row('arm1', 'lstm_own', f'{clan}:fold{k}', candidate, task, result))
    if set(own) != set(ctx.ids):
        raise ValueError('Arm 1 must score every coda exactly once with its own clan')
    for source in ctx.clans:
        models = fold_models[source]
        other = models[0][0]
        if any(not np.array_equal(m[0], other) for m in models):
            raise ValueError(f'{source}: fold models scored different codas')
        ll = np.mean([m[1][0] for m in models], axis=0)
        eos = np.mean([m[1][1] for m in models], axis=0)
        matrix_rows += score_rows(ctx, other, ll, eos, f'arm1:{source}:fold_models', transfer_model_name(source),
                                  source, f'mean of {len(models)} fold models')
    stage.csv('selection.csv', SELECTION_COLUMNS, selection)
    stage.csv('refits.csv', REFIT_COLUMNS, refit_rows)
    stage.csv('own_scores.csv', SCORE_COLUMNS, sorted(own_rows, key=lambda r: (r['target_clan'], r['rq3_coda_id'])))
    stage.csv('matrix_scores.csv', SCORE_COLUMNS,
              sorted(matrix_rows, key=lambda r: (r['train_clans'], r['target_clan'], r['rq3_coda_id'])))
    return own, reproducible, grid


def ngram_self_scores(ctx):
    rows = read_csv(ctx.path('ngram_self_scores_input'), None)
    return {row['rq3_coda_id']: (float(row['bits_per_coda']), int(row['token_count'])) for row in rows}


def gate1_rows(ctx, own):
    """Per clan: own-clan LSTM - own-clan n-gram bits per symbol on the same codas, group-bootstrap CI.
    A clan fails when the CI lies entirely below zero."""
    ngram = ngram_self_scores(ctx)
    group_of = dict(zip(ctx.ids, ctx.group))
    rows = []
    for clan in ctx.clans:
        ids = [ctx.ids[i] for i in ctx.clan_indices(clan)]
        c = paired_contrast({i: own[i] for i in ids}, {i: ngram[i] for i in ids}, group_of,
                            ctx.config['bootstrap_draws'], ctx.config['bootstrap_seed'], ('gate1', clan))
        rows.append(dict(clan=clan, n_codas=c['n_codas'], n_groups=c['n_groups'],
                         lstm_bits_per_symbol=c['first_bits_per_symbol'],
                         ngram_bits_per_symbol=c['second_bits_per_symbol'],
                         difference_bits_per_symbol=c['contrast_bits_per_symbol'],
                         ci95_lower_bits_per_symbol=c['ci95_lower_bits_per_symbol'],
                         ci95_upper_bits_per_symbol=c['ci95_upper_bits_per_symbol'], passed=not ci_below_zero(c)))
    return rows


def stage_arm1(ctx, git):
    device, workers = execution(ctx)
    set_cuda_environment(ctx, device)
    outcome = {}
    with FitPool(ctx.encoded, workers, device) as pool:
        for name, extended in (('arm1', False), ('arm1_rerun', True)):
            stage = Stage(ctx, name, git)
            own, reproducible, grid = run_arm1(ctx, stage, pool, device, extended)
            gate = gate1_rows(ctx, own)
            stage.csv('gate1.csv', GATE1_COLUMNS, gate)
            failed = [row['clan'] for row in gate if not row['passed']]
            stage.finish(device=device, workers=workers, extended_grid=extended, grid=[c['name'] for c in grid],
                         gate_passed=not failed, gate_failed_clans=failed, refit_reproducible=reproducible)
            for row in gate:
                print(f"  gate 1 {row['clan']}: LSTM - n-gram {row['difference_bits_per_symbol']:+.4f} "
                      f"[{row['ci95_lower_bits_per_symbol']:+.4f}, {row['ci95_upper_bits_per_symbol']:+.4f}] "
                      f"{'pass' if row['passed'] else 'FAIL'}", flush=True)
            outcome[name] = failed
            if not failed:
                break
            print(f'{name}: gate 1 fails for {failed}', flush=True)
    if outcome.get('arm1_rerun') and not ctx.config['smoke_test']:
        raise SystemExit(f"Gate 1 still fails after the rerun for {outcome['arm1_rerun']}: stop and fix the fit")
    return outcome


# ---------------------------------------------------------------- arms 2 and 3

ARM_TAGS = (('arm2', True), ('arm3', False))


def pooled_model_name(arm, alpha, hidden_multiplier=1):
    base = f"lstm_pooled_{'tagged' if arm == 'arm2' else 'untagged'}_alpha{alpha}"
    return base if hidden_multiplier == 1 else f'{base}_hidden_x{hidden_multiplier}'


def clan_average(ctx, per_clan):
    """Clans weighted equally: the mean over clans of each clan's bits per symbol."""
    values = [per_clan[clan][0] / per_clan[clan][1] for clan in ctx.clans]
    return sum(values) / len(values)


def stage_arm23(ctx, git):
    device, workers = execution(ctx)
    set_cuda_environment(ctx, device)
    config, model = ctx.config, ctx.config['model']
    stage = Stage(ctx, 'arm23', git)
    grid = candidate_grid(config)
    runs = pooled_runs(ctx)
    plans = {run: pooled_plan(ctx, run) for run in runs}
    tasks, owners = [], []
    for arm, tagged in ARM_TAGS:
        for alpha in config['pooled_alphas']:
            for run in runs:
                plan = plans[run]
                for candidate in grid:
                    tasks.append(make_task(ctx, (arm, alpha, run, candidate['name']), train=plan['select_train'],
                                           val=plan['val'], guard=plan['held'], tagged=tagged, alpha=alpha,
                                           candidate=candidate, seed=model['selection_seed'], epochs=None,
                                           device=device))
                    owners.append((arm, alpha, run, candidate))
    with FitPool(ctx.encoded, workers, device) as pool:
        print(f'arm23: {len(tasks)} selection fits', flush=True)
        results = pool.run(tasks)
        stage.record(results)
        by_owner = {(o[0], o[1], o[2], o[3]['name']): (t, r) for o, t, r in zip(owners, tasks, results)}
        pair_rows = []
        for alpha in config['pooled_alphas']:
            for run in runs:
                for candidate in grid:
                    (ta, ra), (tb, rb) = (by_owner['arm2', alpha, run, candidate['name']],
                                          by_owner['arm3', alpha, run, candidate['name']])
                    common = check_same_draws(f'tag pair {alpha}/{run}/{candidate["name"]}', ta, ra, tb, rb, {'tagged'})
                    pair_rows.append(dict(kind='selection', alpha=alpha, run=run, candidate=candidate['name'],
                                          seed=ta['seed'], epochs_compared=common, identical_draws=True,
                                          identical_initial_weights=ra['init_sha256'] == rb['init_sha256']))
        selection, inner, chosen = [], [], {}
        for arm, tagged in ARM_TAGS:
            for alpha in config['pooled_alphas']:
                for run in runs:
                    idx = [i for i, o in enumerate(owners) if o[:3] == (arm, alpha, run)]
                    best = choose([owners[i][3] for i in idx], [results[i] for i in idx])
                    chosen[arm, alpha, run] = (owners[idx[best]][3], results[idx[best]])
                    require_trained(f'{arm} alpha {alpha} run {run}', results[idx[best]])
                    name = pooled_model_name(arm, alpha)
                    for j, i in enumerate(idx):
                        selection.append(selection_row(arm, name, run, owners[i][3], tasks[i], results[i], j == best))
                        inner += inner_rows(ctx, arm, name, run, owners[i][3], results[i])
        refit_tasks, refit_owners = [], []
        for arm, tagged in ARM_TAGS:
            for alpha in config['pooled_alphas']:
                for run in runs:
                    candidate, result = chosen[arm, alpha, run]
                    plan = plans[run]
                    for seed in model['refit_seeds']:
                        refit_tasks.append(make_task(ctx, (arm, alpha, run, candidate['name'], 'refit'),
                                                     train=plan['refit_train'], val=None, guard=plan['held'],
                                                     score=plan['held'], tagged=tagged, alpha=alpha,
                                                     candidate=candidate, seed=seed, epochs=result['best_epoch'],
                                                     device=device))
                        refit_owners.append((arm, alpha, run, candidate))
        print(f'arm23: {len(refit_tasks)} refits', flush=True)
        refits = pool.run(refit_tasks)
        stage.record(refits)
        reproducible = check_refit_reproducible(pool, refit_tasks[0], refits[0], 'arm23')
    for task, result in zip(refit_tasks, refits):
        check_scored(ctx, ':'.join(map(str, task['key'])), task, result)
    refit_by = {(o[0], o[1], o[2], t['seed']): (o[3], t, r) for o, t, r in zip(refit_owners, refit_tasks, refits)}
    for alpha in config['pooled_alphas']:
        for run in runs:
            for seed in model['refit_seeds']:
                (ca, ta, ra), (cb, tb, rb) = refit_by['arm2', alpha, run, seed], refit_by['arm3', alpha, run, seed]
                if ca['name'] == cb['name']:
                    common = check_same_draws(f'tag refit {alpha}/{run}/{seed}', ta, ra, tb, rb, {'tagged', 'epochs'})
                    pair_rows.append(dict(kind='refit', alpha=alpha, run=run, candidate=ca['name'], seed=seed,
                                          epochs_compared=common, identical_draws=True,
                                          identical_initial_weights=ra['init_sha256'] == rb['init_sha256']))
    rows = {arm: [] for arm, _ in ARM_TAGS}
    refit_rows = []
    for arm, tagged in ARM_TAGS:
        for alpha in config['pooled_alphas']:
            name = pooled_model_name(arm, alpha)
            scored = set()
            for run in runs:
                plan = plans[run]
                group = [refit_by[arm, alpha, run, seed] for seed in model['refit_seeds']]
                ll, eos = mean_over_seeds([r for _, _, r in group])
                rows[arm] += score_rows(ctx, plan['held'], ll, eos, f'{arm}:alpha{alpha}:run{run}', name,
                                        '+'.join(plan['clans']), f'all but fold {run}')
                scored.update(plan['held'].tolist())
                for candidate, task, result in group:
                    refit_rows.append(refit_row(arm, name, run, candidate, task, result))
            if scored != set(range(len(ctx.ids))):
                raise ValueError(f'{name}: the pooled runs must score every coda exactly once')
    # Primary alpha per arm: clan-averaged inner-validation score of each run's chosen fit; a clan's
    # inner-validation codas pooled over the runs in which it has a held-out fold (each coda once).
    alpha_rows, primary = [], {}
    for arm, _ in ARM_TAGS:
        scores = {}
        for alpha in config['pooled_alphas']:
            per_clan = {clan: [0.0, 0] for clan in ctx.clans}
            for run in runs:
                _, result = chosen[arm, alpha, run]
                for code, (ll, n) in result['inner_validation']['by_clan'].items():
                    clan = ctx.clans[code]
                    if run < ctx.n_folds[clan]:
                        per_clan[clan][0] += ll
                        per_clan[clan][1] += n
            for clan in ctx.clans:
                if per_clan[clan][1] != int(ctx.n_symbols[ctx.clan_indices(clan)].sum()):
                    raise ValueError(f'{arm} alpha {alpha}: inner validation must cover every {clan} coda once')
                alpha_rows.append(dict(arm=arm, alpha=alpha, clan=clan, ll_bits=per_clan[clan][0],
                                       n_symbols=per_clan[clan][1],
                                       bits_per_symbol=per_clan[clan][0] / per_clan[clan][1]))
            scores[alpha] = clan_average(ctx, per_clan)
            alpha_rows.append(dict(arm=arm, alpha=alpha, clan='clan_average', ll_bits='', n_symbols='',
                                   bits_per_symbol=scores[alpha]))
        primary[arm] = max(config['pooled_alphas'], key=lambda a: (scores[a], -config['pooled_alphas'].index(a)))
        print(f'{arm}: primary alpha {primary[arm]} (clan-averaged inner validation '
              + ', '.join(f'{a}: {s:.4f}' for a, s in scores.items()) + ')', flush=True)
    stage.csv('selection.csv', SELECTION_COLUMNS, selection)
    stage.csv('inner_validation_by_clan.csv', INNER_COLUMNS, inner)
    stage.csv('refits.csv', REFIT_COLUMNS, refit_rows)
    stage.csv('tag_pairs.csv', ['kind', 'alpha', 'run', 'candidate', 'seed', 'epochs_compared', 'identical_draws',
                                'identical_initial_weights'], pair_rows)
    stage.csv('alpha_scores.csv', ['arm', 'alpha', 'clan', 'll_bits', 'n_symbols', 'bits_per_symbol'], alpha_rows)
    for arm, _ in ARM_TAGS:
        stage.csv(f'{arm}_scores.csv', SCORE_COLUMNS,
                  sorted(rows[arm], key=lambda r: (r['model'], r['target_clan'], r['rq3_coda_id'])))
    return stage.finish(device=device, workers=workers, primary_alpha=primary, refit_reproducible=reproducible)


# ---------------------------------------------------------------- capacity check

def own_scores_of(ctx):
    path = output_dir(ctx, arm1_source(ctx)) / 'own_scores.csv'
    return path, read_scores(ctx, path)['lstm_own']


def stage_capacity(ctx, git):
    device, workers = execution(ctx)
    set_cuda_environment(ctx, device)
    config, model = ctx.config, ctx.config['model']
    stage = Stage(ctx, 'capacity', git)
    own_path, own = own_scores_of(ctx)
    arm2_path = output_dir(ctx, 'arm23') / 'arm2_scores.csv'
    selection_path = output_dir(ctx, 'arm23') / 'selection.csv'
    stage.add_inputs(own_path, arm2_path, selection_path)
    arm2 = read_scores(ctx, arm2_path)
    group_of = dict(zip(ctx.ids, ctx.group))
    trigger_rows, triggered = [], []
    for alpha in config['pooled_alphas']:
        pooled = arm2[pooled_model_name('arm2', alpha)]
        for clan in ctx.clans:
            ids = [ctx.ids[i] for i in ctx.clan_indices(clan)]
            c = paired_contrast({i: pooled[i] for i in ids}, {i: own[i] for i in ids}, group_of,
                                config['bootstrap_draws'], config['bootstrap_seed'], ('pooling_gain', alpha, clan))
            below = ci_below_zero(c)
            trigger_rows.append(dict(alpha=alpha, clan=clan, pooling_gain_bits_per_symbol=c['contrast_bits_per_symbol'],
                                     ci95_lower_bits_per_symbol=c['ci95_lower_bits_per_symbol'],
                                     ci95_upper_bits_per_symbol=c['ci95_upper_bits_per_symbol'],
                                     reliably_below=below))
            if below and alpha not in triggered:
                triggered.append(alpha)
    stage.csv('trigger.csv', ['alpha', 'clan', 'pooling_gain_bits_per_symbol', 'ci95_lower_bits_per_symbol',
                              'ci95_upper_bits_per_symbol', 'reliably_below'], trigger_rows)
    print(f'capacity check: triggered for alpha {triggered}' if triggered else 'capacity check: not triggered',
          flush=True)
    if not triggered:
        return stage.finish(device=device, workers=workers, triggered_alphas=[])
    multiplier = config['capacity_multiplier']
    chosen = {}
    for row in read_csv(selection_path, SELECTION_COLUMNS):
        if row['arm'] == 'arm2' and row['selected'] == 'True':
            chosen[float(row['alpha']), int(row['run'])] = row
    runs = pooled_runs(ctx)
    plans = {run: pooled_plan(ctx, run) for run in runs}
    tasks, owners = [], []
    for alpha in triggered:
        for run in runs:
            row = chosen[float(alpha), run]
            candidate = dict(hidden=int(row['hidden']) * multiplier, dropout=float(row['dropout']),
                             learning_rate=float(row['learning_rate']), patience=int(row['patience']))
            candidate['name'] = name_of(candidate)
            plan = plans[run]
            tasks.append(make_task(ctx, ('capacity', alpha, run, candidate['name']), train=plan['select_train'],
                                   val=plan['val'], guard=plan['held'], tagged=True, alpha=alpha, candidate=candidate,
                                   seed=model['selection_seed'], epochs=None, device=device))
            owners.append((alpha, run, candidate))
    with FitPool(ctx.encoded, workers, device) as pool:
        results = pool.run(tasks)
        stage.record(results)
        refit_tasks, refit_owners, selection = [], [], []
        for (alpha, run, candidate), task, result in zip(owners, tasks, results):
            require_trained(f'capacity alpha {alpha} run {run}', result)
            name = pooled_model_name('arm2', alpha, multiplier)
            selection.append(selection_row('capacity', name, run, candidate, task, result, True))
            plan = plans[run]
            for seed in model['refit_seeds']:
                refit_tasks.append(make_task(ctx, ('capacity', alpha, run, candidate['name'], 'refit'),
                                             train=plan['refit_train'], val=None, guard=plan['held'],
                                             score=plan['held'], tagged=True, alpha=alpha, candidate=candidate,
                                             seed=seed, epochs=result['best_epoch'], device=device))
                refit_owners.append((alpha, run, candidate))
        refits = pool.run(refit_tasks)
        stage.record(refits)
        reproducible = check_refit_reproducible(pool, refit_tasks[0], refits[0], 'capacity')
    rows, refit_rows = [], []
    for alpha in triggered:
        name = pooled_model_name('arm2', alpha, multiplier)
        for run in runs:
            group = [(o, t, r) for o, t, r in zip(refit_owners, refit_tasks, refits) if o[:2] == (alpha, run)]
            for _, t, r in group:
                check_scored(ctx, name, t, r)
                refit_rows.append(refit_row('capacity', name, run, group[0][0][2], t, r))
            ll, eos = mean_over_seeds([r for _, _, r in group])
            plan = plans[run]
            rows += score_rows(ctx, plan['held'], ll, eos, f'capacity:alpha{alpha}:run{run}', name,
                               '+'.join(plan['clans']), f'all but fold {run}')
    stage.csv('selection.csv', SELECTION_COLUMNS, selection)
    stage.csv('refits.csv', REFIT_COLUMNS, refit_rows)
    stage.csv('scores.csv', SCORE_COLUMNS, sorted(rows, key=lambda r: (r['model'], r['target_clan'], r['rq3_coda_id'])))
    return stage.finish(device=device, workers=workers, triggered_alphas=triggered, hidden_multiplier=multiplier,
                        refit_reproducible=reproducible)


# ---------------------------------------------------------------- arm 4

def checkpoint_path(ctx, left_out, seed):
    return ctx.root / ctx.config['checkpoint_directory'] / f'arm4_without_{left_out}_seed{seed}.pt'


def stage_arm4(ctx, git):
    device, workers = execution(ctx)
    set_cuda_environment(ctx, device)
    config, model = ctx.config, ctx.config['model']
    stage = Stage(ctx, 'arm4', git)
    arm23_summary = output_dir(ctx, 'arm23') / 'summary.json'
    stage.add_inputs(arm23_summary)
    alpha = read_json(arm23_summary)['primary_alpha']['arm3']
    grid = candidate_grid(config)
    plans = {clan: loo_plan(ctx, clan) for clan in ctx.clans}
    tasks, owners = [], []
    for clan, plan in plans.items():
        for candidate in grid:
            tasks.append(make_task(ctx, ('arm4', clan, candidate['name']), train=plan['select_train'], val=plan['val'],
                                   guard=plan['held'], tagged=False, alpha=alpha, candidate=candidate,
                                   seed=model['selection_seed'], epochs=None, device=device))
            owners.append((clan, candidate))
    (ctx.root / config['checkpoint_directory']).mkdir(parents=True, exist_ok=True)
    with FitPool(ctx.encoded, workers, device) as pool:
        print(f'arm4: {len(tasks)} selection fits at alpha {alpha}', flush=True)
        results = pool.run(tasks)
        stage.record(results)
        selection, inner, chosen = [], [], {}
        for clan in ctx.clans:
            idx = [i for i, o in enumerate(owners) if o[0] == clan]
            best = choose([owners[i][1] for i in idx], [results[i] for i in idx])
            chosen[clan] = (owners[idx[best]][1], results[idx[best]])
            require_trained(f'arm4 without {clan}', results[idx[best]])
            for j, i in enumerate(idx):
                selection.append(selection_row('arm4', 'lstm_leave_one_out', f'without_{clan}', owners[i][1], tasks[i],
                                               results[i], j == best))
                inner += inner_rows(ctx, 'arm4', 'lstm_leave_one_out', f'without_{clan}', owners[i][1], results[i])
        refit_tasks = []
        for clan, plan in plans.items():
            candidate, result = chosen[clan]
            for seed in model['refit_seeds']:
                refit_tasks.append(make_task(ctx, ('arm4', clan, candidate['name'], 'refit'), train=plan['refit_train'],
                                             val=None, guard=plan['held'], score=plan['held'], tagged=False,
                                             alpha=alpha, candidate=candidate, seed=seed, epochs=result['best_epoch'],
                                             device=device, save_checkpoint=checkpoint_path(ctx, clan, seed)))
        print(f'arm4: {len(refit_tasks)} refits', flush=True)
        refits = pool.run(refit_tasks)
        stage.record(refits)
        reproducible = check_refit_reproducible(pool, refit_tasks[0], refits[0], 'arm4')
    rows, refit_rows, checkpoints = [], [], {}
    for clan, plan in plans.items():
        candidate, _ = chosen[clan]
        group = [(t, r) for t, r in zip(refit_tasks, refits) if t['key'][1] == clan]
        for t, r in group:
            check_scored(ctx, f'arm4 without {clan}', t, r)
            check_excludes_clan(f'arm4 without {clan} refit', ctx, clan, t['train_idx'])
            refit_rows.append(refit_row('arm4', 'lstm_leave_one_out', f'without_{clan}', candidate, t, r))
            checkpoints.setdefault(clan, {})[str(t['seed'])] = dict(
                path=Path(r['checkpoint']).resolve().relative_to(ctx.root).as_posix(), state_sha256=r['state_sha256'])
        ll, eos = mean_over_seeds([r for _, r in group])
        rows += score_rows(ctx, plan['held'], ll, eos, f'arm4:without_{clan}', 'lstm_leave_one_out',
                           '+'.join(plan['clans']), 'all')
    stage.csv('selection.csv', SELECTION_COLUMNS, selection)
    stage.csv('inner_validation_by_clan.csv', INNER_COLUMNS, inner)
    stage.csv('refits.csv', REFIT_COLUMNS, refit_rows)
    stage.csv('scores.csv', SCORE_COLUMNS, sorted(rows, key=lambda r: (r['target_clan'], r['rq3_coda_id'])))
    return stage.finish(device=device, workers=workers, alpha=alpha, refit_reproducible=reproducible,
                        chosen={clan: dict(hidden=c['hidden'], dropout=c['dropout'], learning_rate=c['learning_rate'],
                                           patience=c['patience'], best_epoch=r['best_epoch'])
                                for clan, (c, r) in chosen.items()},
                        checkpoints=checkpoints)


# ---------------------------------------------------------------- arm 5

def share_draw(ctx, pool_idx, share, clan, fold, draw):
    """A share f of the clan's training folds (inner-validation fold excluded), drawn at coda level."""
    m = max(1, round(share * len(pool_idx)))
    rng = random.Random(pair_seed(ctx.config['bootstrap_seed'], 'arm5', clan, str(fold), repr(share), str(draw)))
    return np.sort(np.array(rng.sample(sorted(pool_idx.tolist()), m), dtype=np.int64))


def stage_arm5(ctx, git):
    device, workers = execution(ctx)
    set_cuda_environment(ctx, device)
    config, model = ctx.config, ctx.config['model']
    stage = Stage(ctx, 'arm5', git)
    pilot_summary = output_dir(ctx, 'pilot') / 'summary.json'
    arm4_summary_path = output_dir(ctx, 'arm4') / 'summary.json'
    stage.add_inputs(pilot_summary, arm4_summary_path)
    plan5 = read_json(pilot_summary)['decision']['arm5']
    arm4 = read_json(arm4_summary_path)
    learning_rates = config['arm5']['learning_rates']
    seeds = model['refit_seeds'][:plan5['draws']]
    tasks, owners = [], []
    for clan in ctx.clans:
        settings = arm4['chosen'][clan]
        inherited = loo_plan(ctx, clan)['refit_train']
        for fold in range(ctx.n_folds[clan]):
            plan = own_clan_plan(ctx, clan, fold)
            for share in plan5['shares']:
                for draw, seed in enumerate(seeds):
                    train = share_draw(ctx, plan['select_train'], share, clan, fold, draw)
                    checkpoint = arm4['checkpoints'][clan][str(seed)]
                    for version in ('finetuned', 'scratch'):
                        for lr in learning_rates:
                            candidate = dict(hidden=settings['hidden'], dropout=settings['dropout'], learning_rate=lr,
                                             patience=model['patience'])
                            candidate['name'] = name_of(candidate)
                            finetuned = version == 'finetuned'
                            tasks.append(make_task(
                                ctx, ('arm5', clan, fold, share, draw, version, lr), train=train, val=plan['val'],
                                guard=plan['held'], score=plan['held'], tagged=False, alpha=1.0, candidate=candidate,
                                seed=seed, epochs=None, device=device, inherited=(inherited,) if finetuned else (),
                                init_checkpoint=ctx.root / checkpoint['path'] if finetuned else None,
                                init_checkpoint_sha256=checkpoint['state_sha256'] if finetuned else None))
                            owners.append((clan, fold, share, draw, version, lr, candidate, plan))
    with FitPool(ctx.encoded, workers, device) as pool:
        print(f'arm5: {len(tasks)} fits (shares {plan5["shares"]}, {plan5["draws"]} draw(s))', flush=True)
        results = pool.run(tasks)
        stage.record(results)
        reproducible = check_refit_reproducible(pool, tasks[0], results[0], 'arm5')
    by = {o[:6]: (o, t, r) for o, t, r in zip(owners, tasks, results)}
    selection, group_rows, pair_rows, epoch0 = [], [], [], 0
    for clan in ctx.clans:
        for fold in range(ctx.n_folds[clan]):
            for share in plan5['shares']:
                for draw in range(len(seeds)):
                    for lr in learning_rates:
                        (_, ta, ra), (_, tb, rb) = (by[clan, fold, share, draw, 'finetuned', lr],
                                                    by[clan, fold, share, draw, 'scratch', lr])
                        common = check_same_draws(f'arm5 pair {clan}/{fold}/{share}/{draw}/{lr}', ta, ra, tb, rb,
                                                  {'init_checkpoint', 'init_checkpoint_sha256'})
                        pair_rows.append(dict(clan=clan, fold=fold, share=share, draw=draw, learning_rate=lr,
                                              epochs_compared=common, identical_draws=True))
                    for version in ('finetuned', 'scratch'):
                        group = [by[clan, fold, share, draw, version, lr] for lr in learning_rates]
                        best = max(range(len(group)), key=lambda i: (
                            group[i][2]['inner_validation']['bits_per_symbol'], -i))
                        for i, (o, t, r) in enumerate(group):
                            check_scored(ctx, ':'.join(map(str, t['key'])), t, r)
                            selection.append(selection_row('arm5', f'{version}_f{share}_draw{draw}',
                                                           f'{clan}:fold{fold}', o[6], t, r, i == best))
                        o, t, r = group[best]
                        epoch0 += r['best_epoch'] == 0
                        plan = o[7]
                        held = plan['held']
                        sums = {}
                        for i, coda in enumerate(held):
                            s = sums.setdefault(ctx.group[coda], [0, 0, 0.0, 0.0])
                            s[0] += 1
                            s[1] += int(r['scores']['n_symbols'][i])
                            s[2] += float(r['scores']['ll_bits'][i])
                            s[3] += float(r['scores']['ll_eos_bits'][i])
                        for group_id, (n_codas, n_symbols, ll, eos) in sorted(sums.items()):
                            expected = sum(ctx.vlad_token_count[ctx.ids[c]] for c in held if ctx.group[c] == group_id)
                            if n_symbols != expected:
                                raise ValueError(f'arm5 {clan}/{group_id}: n_symbols differ from Vlad token_count')
                            group_rows.append(dict(
                                run_id=f'arm5:{clan}:fold{fold}:f{share}:draw{draw}', model=f'lstm_{version}',
                                tokeniser=config['tokenizer_id'], train_clans=clan,
                                train_folds=f'share {share} of folds {fold_label(sorted(set(ctx.fold_of[t["train_idx"]].tolist())))}',
                                target_clan=clan, target_fold=fold, share=share, draw=draw, version=version,
                                learning_rate=t['learning_rate'], best_epoch=r['best_epoch'], group_id=group_id,
                                n_codas=n_codas, n_symbols=n_symbols, ll_bits_sum=ll, ll_eos_bits_sum=eos))
    stage.csv('selection.csv', SELECTION_COLUMNS, selection)
    stage.csv('group_sums.csv', GROUP_SUM_COLUMNS, group_rows)
    stage.csv('pairs.csv', ['clan', 'fold', 'share', 'draw', 'learning_rate', 'epochs_compared', 'identical_draws'],
              pair_rows)
    return stage.finish(device=device, workers=workers, shares=plan5['shares'], draws=plan5['draws'],
                        reduced=plan5['reduced'], chosen_at_epoch0=epoch0, refit_reproducible=reproducible)


# ---------------------------------------------------------------- CLI

def run_stage(config_path, stage):
    ctx = Context(config_path)
    git = git_state()
    check_tree(ctx, git)
    started = time.time()
    if stage == 'report':
        from .report import stage_report
        result = stage_report(ctx, git)
    else:
        result = dict(pilot=stage_pilot, arm1=stage_arm1, arm23=stage_arm23, capacity=stage_capacity,
                      arm4=stage_arm4, arm5=stage_arm5)[stage](ctx, git)
    print(f'{stage} finished in {time.time() - started:.0f} s', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='config/rq3_neural.json')
    parser.add_argument('--stage', required=True, choices=STAGES + ('all',))
    args = parser.parse_args(argv)
    for stage in (STAGES if args.stage == 'all' else (args.stage,)):
        run_stage(args.config, stage)


if __name__ == '__main__':
    main()
