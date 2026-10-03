"""Checks that what is committed is what the committed code and inputs produce.

- Preprocessing regenerates byte-identical files (catches stale local inputs, line-ending
  drift and hand-edited generated reports).
- Every recorded SHA-256 (fold manifests, result summaries) matches the committed file.
- The committed fold manifests keep test, validation and training days apart.

Standard library only; the regeneration runs in a temporary copy, never in the repository.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = sorted((ROOT / 'data' / 'processed').glob('*/manifest.json'))
SUMMARIES = sorted((ROOT / 'data' / 'results').glob('**/summary.json'))

# Preprocessing commands whose outputs are committed, in pipeline order.
PREPROCESSING = [
    ('preprocessing.prepare', 'config/rq1.json'),
    ('preprocessing.prepare_rq2', 'config/rq2.json'),
    ('preprocessing.folds', 'config/rq1_folds.json'),
    ('preprocessing.folds', 'config/rq1_folds_retrospective.json'),
    ('preprocessing.inventories', 'config/rq2_inventories.json'),
    ('preprocessing.folds', 'config/rq2_b_dtree_folds.json'),
    ('preprocessing.folds', 'config/rq2_c_rhythm_folds.json'),
    ('preprocessing.folds', 'config/rq2_d_rhythm_tempo_folds.json'),
    ('preprocessing.folds', 'config/rq2_e_type_tempo_folds.json'),
    ('preprocessing.folds', 'config/rq2_d_rhythm_tempo_tdict_folds.json'),
    ('preprocessing.folds', 'config/rq2_e_type_tempo_tdict_folds.json'),
]

# Result summaries fitted before the CRLF provenance fix (2026-09-28). Their inputs were the
# committed files except for one provenance field: rq1_targets.jsonl rows carried source_sha256 of
# the raw CSV read with CRLF line endings, and the fold manifests built from that file record its
# hash. The test does not trust this list: for each listed summary it rebuilds that variant of the
# committed inputs and requires the recorded hash to match it exactly, so any real change to the
# committed targets or manifests still fails. Remove an entry when that result is regenerated; the
# test fails for entries that are no longer needed, so this list can only shrink.
CRLF_PROVENANCE = {
    'data/results/rq1/evaluation/summary.json',
    'data/results/rq1_logistic_v2/evaluation/summary.json',
    'data/results/rq1_logistic_v2_retrospective/evaluation/summary.json',
    'data/results/rq1_lstm/evaluation/summary.json',
    'data/results/rq1_lstm_reproduction/evaluation/summary.json',
}


def crlf_provenance_variant(path):
    """SHA-256 of a committed input as it was before the fix: raw-CSV hash computed from CRLF
    line endings, and (in manifests) the hash of the targets file carrying that value."""
    raw = (ROOT / read_json(ROOT / 'config' / 'preprocessing.json')['input']).read_bytes()
    lf_raw = hashlib.sha256(raw).hexdigest().encode()
    crlf_raw = hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest().encode()
    targets = (ROOT / 'data' / 'processed' / 'rq1_targets.jsonl').read_bytes()
    old_targets = hashlib.sha256(targets.replace(lf_raw, crlf_raw)).hexdigest().encode()
    content = Path(path).read_bytes().replace(lf_raw, crlf_raw)
    content = content.replace(hashlib.sha256(targets).hexdigest().encode(), old_targets)
    return hashlib.sha256(content).hexdigest()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def files_under(base):
    return {p.relative_to(base).as_posix(): p for p in base.rglob('*') if p.is_file()}


class RegenerationTests(unittest.TestCase):
    def test_preprocessing_regenerates_committed_files(self):
        compared = ('data/raw', 'data/interim', 'data/processed', 'reports')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copytree(ROOT / 'config', tmp / 'config')
            for rel in compared:
                shutil.copytree(ROOT / rel, tmp / rel)
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
            for module, config in PREPROCESSING:
                result = subprocess.run([sys.executable, '-m', f'whale_coda_dynamics.{module}', '--config', config],
                                        cwd=tmp, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f'{module} {config} failed:\n{result.stderr[-4000:]}')
            for rel in compared:
                before, after = files_under(ROOT / rel), files_under(tmp / rel)
                self.assertEqual(sorted(after), sorted(before), f'{rel}: pipeline created or removed files')
                changed = [name for name in before if before[name].read_bytes() != after[name].read_bytes()]
                self.assertEqual(changed, [], f'{rel}: regenerated files differ from the committed ones. '
                                              'Commit the regenerated files (and the code that produced them) together.')


class RecordedHashTests(unittest.TestCase):
    def test_fold_manifests_match_committed_inputs(self):
        self.assertTrue(MANIFESTS)
        raw = read_json(ROOT / 'config' / 'preprocessing.json')['input']
        for manifest_path in MANIFESTS:
            manifest = read_json(manifest_path)
            configuration = manifest['configuration']
            with self.subTest(manifest=manifest_path.relative_to(ROOT).as_posix()):
                self.assertEqual(manifest['input_sha256'], sha256(ROOT / configuration['input']))
                self.assertEqual(manifest['preparation_config_sha256'], sha256(ROOT / configuration['preparation_config']))
                self.assertEqual(manifest['source_sha256'], sha256(ROOT / raw))

    def test_result_summaries_match_committed_inputs(self):
        self.assertTrue(SUMMARIES)
        still_old = set()
        for summary_path in SUMMARIES:
            summary_rel = summary_path.relative_to(ROOT).as_posix()
            for recorded_path, recorded_hash in read_json(summary_path).get('input_sha256', {}).items():
                path = recorded_path.replace('\\', '/')  # older summaries were written on Windows
                with self.subTest(summary=summary_rel, input=path):
                    self.assertTrue((ROOT / path).is_file(), 'recorded input no longer exists')
                    if recorded_hash == sha256(ROOT / path):
                        continue
                    still_old.add(summary_rel)
                    self.assertIn(summary_rel, CRLF_PROVENANCE,
                                  'result was produced from inputs that differ from the committed ones; '
                                  'rerun it from a clean checkout')
                    self.assertEqual(recorded_hash, crlf_provenance_variant(ROOT / path),
                                     'inputs differ from the committed ones beyond the known CRLF provenance field')
        self.assertEqual(sorted(CRLF_PROVENANCE - still_old), [],
                         'these results now match main: remove them from CRLF_PROVENANCE')


class FoldManifestTests(unittest.TestCase):
    def test_days_and_targets_never_cross_roles(self):
        targets = {}
        for line in (ROOT / 'data' / 'processed' / 'rq1_targets.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            targets[row['coda_id']] = row
        eligible = {c for c, r in targets.items() if r['comparison_eligible']}
        for manifest_path in MANIFESTS:
            manifest = read_json(manifest_path)
            days = {r['recording_date'] for r in targets.values()}
            test_days, validation_days = Counter(), Counter()
            with self.subTest(manifest=manifest_path.relative_to(ROOT).as_posix()):
                self.assertEqual(len(manifest['folds']), len(days))
                tested = []
                for fold in manifest['folds']:
                    roles = fold['day_splits']
                    self.assertEqual(set(roles), days)
                    self.assertEqual(Counter(roles.values()), Counter(test=1, validation=1, train=len(days) - 2))
                    self.assertEqual(roles[fold['test_day']], 'test')
                    self.assertEqual(roles[fold['validation_day']], 'validation')
                    test_days[fold['test_day']] += 1
                    validation_days[fold['validation_day']] += 1
                    ids = fold['comparison_ids']
                    for role in ('train', 'validation', 'test'):
                        for coda_id in ids[role]:
                            self.assertEqual(roles[targets[coda_id]['recording_date']], role, coda_id)
                    self.assertFalse(set(ids['train']) & set(ids['validation']))
                    self.assertFalse(set(ids['train']) & set(ids['test']))
                    self.assertFalse(set(ids['validation']) & set(ids['test']))
                    tested += ids['test']
                self.assertEqual(test_days, Counter({d: 1 for d in days}))
                self.assertEqual(validation_days, Counter({d: 1 for d in days}))
                self.assertEqual(sorted(tested), sorted(eligible), 'every comparison target is tested exactly once')


if __name__ == '__main__':
    unittest.main()
