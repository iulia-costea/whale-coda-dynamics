"""Shared paired RQ1 evaluation in bits per coda; no model fitting.

Prediction JSONL rows: coda_id, fold_id, condition, seed (integer), probabilities
(mapping from every fold-vocabulary label to a strictly positive probability).
Ground truth, day, vocabulary and own-last label come from prepared data/manifest,
never from predictor-supplied metadata. CLI input must cover all manifest conditions
and exactly the configured seeds. Neural runs should configure three seeds.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text


def log_likelihood_bits(probability):
    if type(probability) not in (float, int) or not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("Likelihood requires a finite probability in (0, 1]; zero is not silently clipped")
    return math.log2(probability)


def repeat_switch_bits(probabilities, target, own_last):
    """Exact decomposition in the mapped vocabulary, including UNK merges."""
    total = log_likelihood_bits(probabilities[target])
    if own_last is None:
        return None, None
    if target == own_last:
        return total, 0.0
    # Sum alternatives rather than subtracting from one near the boundary.
    switch_mass = math.fsum(p for label, p in probabilities.items() if label != own_last)
    repeat_part = log_likelihood_bits(switch_mass)
    return repeat_part, total - repeat_part


def classification_metrics(targets, predicted):
    """Macro-F1 over the union of observed true and predicted labels."""
    if not targets or len(targets) != len(predicted):
        raise ValueError("Require aligned nonempty labels")
    labels = sorted(set(targets) | set(predicted))
    f1 = []
    for label in labels:
        tp = sum(a == b == label for a, b in zip(targets, predicted))
        fp = sum(a != label and b == label for a, b in zip(targets, predicted))
        fn = sum(a == label and b != label for a, b in zip(targets, predicted))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return {"accuracy": sum(a == b for a, b in zip(targets, predicted)) / len(targets),
            "macro_f1": math.fsum(f1) / len(f1), "macro_f1_labels": labels}


def day_bootstrap_bits(day_values, replicates=2000, seed=42, confidence=0.95):
    """Resample whole days; duplicate draws duplicate all that day's tokens."""
    if type(replicates) is not int or replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    days = sorted(day_values)
    if len(days) < 2 or any(not day_values[d] for d in days):
        raise ValueError("Bootstrap needs at least two nonempty days")
    if any(not math.isfinite(v) for d in days for v in day_values[d]):
        raise ValueError("Nonfinite contrast")
    sums = {d: math.fsum(day_values[d]) for d in days}
    sizes = {d: len(day_values[d]) for d in days}
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = rng.choices(days, k=len(days))
        samples.append(math.fsum(sums[d] for d in selected) / sum(sizes[d] for d in selected))
    samples.sort()
    def quantile(q):
        position = q * (len(samples) - 1)
        lo = math.floor(position)
        hi = math.ceil(position)
        return samples[lo] + (position - lo) * (samples[hi] - samples[lo])
    tail = (1 - confidence) / 2
    return {"mean_bits_per_coda": math.fsum(sums.values()) / sum(sizes.values()),
            "ci_bits_per_coda": [quantile(tail), quantile(1 - tail)],
            "positive_days": sum(sums[d] > 0 for d in days), "days": len(days),
            "day_means_bits": {d: sums[d] / sizes[d] for d in days}}


def evaluate(predictions, manifest, targets, seeds, replicates=2000, bootstrap_seed=42, confidence=0.95):
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide unique integer model seeds")
    rows = {r['coda_id']: r for r in targets}
    if len(rows) != len(targets):
        raise ValueError("Duplicate prepared targets")
    conditions = list(manifest['input_conditions'])
    expected, metadata = set(), {}
    for fold in manifest['folds']:
        vocabulary = fold['vocabulary_policy']['vocabulary']
        known = set(vocabulary)
        unk = fold['vocabulary_policy']['unk_token']
        for coda_id in fold['comparison_ids']['test']:
            if coda_id in metadata:
                raise ValueError("Target assigned to multiple test folds")
            r = rows[coda_id]
            if not r['comparison_eligible'] or r['recording_date'] != fold['test_day']:
                raise ValueError("Manifest test assignment disagrees with prepared data")
            raw = r['Ct_raw']
            last_raw = rows[r['A_history_ids'][-1]]['Ct_raw'] if r['A_history_ids'] else None
            label = raw if raw in known else unk
            last = last_raw if last_raw in known else unk if last_raw is not None else None
            metadata[coda_id] = dict(fold_id=fold['fold_id'], recording_date=r['recording_date'],
                                     target=label, target_raw=raw, own_last=last,
                                     raw_repeat=raw == last_raw if last_raw is not None else None,
                                     mapped_repeat=label == last if last is not None else None,
                                     vocabulary=vocabulary)
            expected.update((coda_id, fold['fold_id'], c, s) for c in conditions for s in seeds)
    if set(metadata) != {r['coda_id'] for r in targets if r['comparison_eligible']}:
        raise ValueError("Manifest does not cover all comparison targets")
    scored, seen = [], set()
    for p in predictions:
        key = (p['coda_id'], p['fold_id'], p['condition'], p['seed'])
        if type(p['seed']) is not int or key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate prediction")
        seen.add(key)
        meta = metadata[p['coda_id']]
        probs = p['probabilities']
        if set(probs) != set(meta['vocabulary']):
            raise ValueError("Prediction vocabulary differs from fold vocabulary")
        for v in probs.values():
            log_likelihood_bits(v)
        if not math.isclose(math.fsum(probs.values()), 1, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Probabilities must sum to one")
        repetition, choice = repeat_switch_bits(probs, meta['target'], meta['own_last'])
        ll = log_likelihood_bits(probs[meta['target']])
        if repetition is not None and not math.isclose(repetition + choice, ll, abs_tol=1e-12):
            raise ValueError("Repeat/switch decomposition does not sum to likelihood")
        scored.append(dict(coda_id=p['coda_id'], condition=p['condition'], seed=p['seed'],
                           **{k: v for k, v in meta.items() if k != 'vocabulary'},
                           predicted=max(sorted(probs), key=probs.get), log_likelihood_bits=ll,
                           repeat_bits=repetition, switch_choice_bits=choice))
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} predictions")
    scored.sort(key=lambda r: (r['coda_id'], r['condition'], r['seed']))
    grouped = defaultdict(list)
    for r in scored:
        grouped[r['condition'], r['coda_id']].append(r)
    averages = {}
    for key, rs in grouped.items():
        averages[key] = {f: math.fsum(r[f] for r in rs) / len(rs) if rs[0][f] is not None else None
                         for f in ('log_likelihood_bits', 'repeat_bits', 'switch_choice_bits')}
    metrics = {}
    for condition in conditions:
        per_seed = {}
        for seed in seeds:
            rs = [r for r in scored if r['condition'] == condition and r['seed'] == seed]
            per_seed[str(seed)] = dict(classification_metrics([r['target'] for r in rs], [r['predicted'] for r in rs]),
                                      mean_log_likelihood_bits=math.fsum(r['log_likelihood_bits'] for r in rs) / len(rs))
        metrics[condition] = {"per_seed": per_seed,
                              **{f: math.fsum(m[f] for m in per_seed.values()) / len(seeds)
                                 for f in ('accuracy', 'macro_f1', 'mean_log_likelihood_bits')}}
    pairs = {'own_history': ('A_B_S', 'B_S'), 'partner_content': ('A_B_S', 'A_S'),
             'timing': ('A_B_S', 'A_B')}
    sources = {c: set(c.split('_')) if c != 'none' else set() for c in conditions}
    for big in conditions:
        for small in conditions:
            if sources[small] < sources[big] and len(sources[big] - sources[small]) == 1:
                pairs[f'add_input:{big}-{small}'] = (big, small)
    # Descriptive comparison requested in M2; not a tuning/selection rule.
    better = max(('A', 'B'), key=lambda c: metrics[c]['mean_log_likelihood_bits'])
    pairs['both_vs_better_single'] = ('A_B', better)
    contrasts, paired = {}, []
    for name, (big, small) in pairs.items():
        components = defaultdict(lambda: defaultdict(list))
        for coda_id, meta in sorted(metadata.items()):
            out = dict(contrast=name, coda_id=coda_id, recording_date=meta['recording_date'])
            for field in ('log_likelihood_bits', 'repeat_bits', 'switch_choice_bits'):
                a, b = averages[big, coda_id][field], averages[small, coda_id][field]
                delta = a - b if a is not None and b is not None else None
                out[field] = delta
                if delta is not None:
                    components[field][meta['recording_date']].append(delta)
            paired.append(out)
        contrasts[name] = {'full': big, 'reduced': small, 'components': {
            f: day_bootstrap_bits(values, replicates, bootstrap_seed, confidence)
            for f, values in components.items()}}
    summary = dict(units='bits per coda', seeds=seeds, comparison_targets=len(metadata),
                   bootstrap=dict(replicates=replicates, seed=bootstrap_seed, confidence=confidence,
                                  method='percentile; days resampled with replacement; token-weighted mean'),
                   classification_rule='metrics computed per seed then averaged; macro-F1 uses observed true/predicted label union',
                   decomposition_rule='mapped vocabulary; all-target contribution, not switch-only conditional mean',
                   raw_repeats=sum(m['raw_repeat'] is True for m in metadata.values()),
                   mapped_repeats=sum(m['mapped_repeat'] is True for m in metadata.values()),
                   decomposition_targets=sum(m['own_last'] is not None for m in metadata.values()),
                   add_input_negative_contrasts=[name for name, value in contrasts.items()
                       if name.startswith('add_input:') and value['components']['log_likelihood_bits']['mean_bits_per_coda'] < 0],
                   better_single_note='A/B selected by aggregate held-out LL for descriptive M2 comparison; CI conditions on that selection',
                   metrics=metrics, contrasts=contrasts)
    return summary, scored, paired


def run(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest_path = root / config['manifest']
    manifest = read_json(manifest_path)
    target_path = root / manifest['configuration']['input']
    if hashlib.sha256(target_path.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError("Fold manifest is stale; regenerate it for the current prepared targets")
    prediction_path = root / config['predictions']
    summary, scored, paired = evaluate(read_jsonl(prediction_path), manifest, read_jsonl(target_path),
                                      config['seeds'], config['bootstrap_replicates'],
                                      config['bootstrap_seed'], config['confidence'])
    summary['configuration'] = config
    summary['input_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (manifest_path, target_path, prediction_path)}
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'scored_predictions.jsonl', scored)
    write_jsonl(output / 'paired_contrasts.jsonl', paired)
    write_text(output / 'summary.json', json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq1_evaluation.json')
    summary = run(parser.parse_args().config)
    print(f"Evaluated {summary['comparison_targets']} paired targets. Units: bits per coda.")


if __name__ == '__main__':
    main()
