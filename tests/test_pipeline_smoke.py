"""End-to-end smoke runs of every model runner, through the report step.

Each runner is started exactly as a person would start it (``python -m ... --config``)
on the full 14-fold manifest, with the smallest settings that still exercise every
code path: one L2 value, one seed, one or two epochs, few bootstrap replicates. The
numbers are meaningless; the point is that a clean checkout can produce a report.
This catches uncommitted helper functions, broken imports and changed signatures.

Everything is written to a temporary directory, never into the repository.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_SCIPY = all(importlib.util.find_spec(m) for m in ('numpy', 'scipy'))
HAS_TORCH = HAS_SCIPY and importlib.util.find_spec('torch') is not None
CONDITIONS = {'none', 'S', 'A', 'B', 'A_B', 'A_S', 'B_S', 'A_B_S'}


class Sandbox:
    """A throwaway project root holding copies of the prepared inputs and smoke configs.

    Runners resolve every path against their config file's parent.parent, so a config
    in <tmp>/config/ reads and writes only inside <tmp>.
    """

    def __init__(self, tmp):
        self.root = Path(tmp)
        for rel in ('data/processed/rq1_targets.jsonl',
                    'data/processed/rq1_folds/manifest.json',
                    'data/processed/rq1_folds_retrospective/manifest.json'):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, self.root / rel)
        (self.root / 'config').mkdir()

    def config(self, name, **values):
        path = self.root / 'config' / name
        path.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
        return path

    def evaluation(self, name, manifest, seeds):
        return self.config(name, manifest=manifest, predictions='unused', output_directory='unused',
                           seeds=seeds, bootstrap_replicates=50, bootstrap_seed=42, confidence=0.95)

    def run(self, module, config):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        result = subprocess.run([sys.executable, '-m', f'whale_coda_dynamics.{module}', '--config', str(config)],
                                cwd=self.root, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f'{module} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}')
        return result.stdout

    def check_outputs(self, test, output_directory, report, conditions=CONDITIONS):
        summary = json.loads((self.root / output_directory / 'evaluation' / 'summary.json').read_text(encoding='utf-8'))
        test.assertEqual(set(summary['metrics']), conditions)
        text = (self.root / report).read_text(encoding='utf-8')
        test.assertGreater(len(text.strip()), 0)
        return summary


PRIMARY = 'data/processed/rq1_folds/manifest.json'
RETROSPECTIVE = 'data/processed/rq1_folds_retrospective/manifest.json'
LOGISTIC_V2 = dict(history_order=3, l2_grid=[0.01], max_iterations=2000, gradient_tolerance=1e-6, seed=42)


@unittest.skipUnless(HAS_SCIPY, 'needs numpy and scipy (pip install ".[modeling]")')
class LogisticSmokeTests(unittest.TestCase):
    def test_v4_runner_and_retrospective_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            box.evaluation('eval_retro.json', RETROSPECTIVE, [42])
            reference = box.config('v3.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                   output_directory='out/v3', report='out/v3.md', **LOGISTIC_V2)
            box.run('logistic', reference)
            settings = dict(LOGISTIC_V2, s_l2_grid=[.1], version_label='v4', compare_label='v3')
            primary = box.config('v4.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                 output_directory='out/v4', report='out/v4.md',
                                 compare_against='out/v3/evaluation/summary.json', **settings)
            box.run('logistic', primary)
            box.check_outputs(self, 'out/v4', 'out/v4.md')
            retro = box.config('v4_retro.json', manifest=RETROSPECTIVE, evaluation_config='config/eval_retro.json',
                               output_directory='out/v4_retro', report='out/v4_retro.md',
                               compare_against='out/v4/evaluation/summary.json',
                               sensitivity='retrospective_delay', **settings)
            box.run('logistic', retro)
            box.check_outputs(self, 'out/v4_retro', 'out/v4_retro.md')

    def test_v1_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            config = box.config('logistic.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                output_directory='out/v1', report='out/v1.md', history_order=3, seed=42,
                                l2=0.01, max_iterations=2000, gradient_tolerance=1e-6)
            box.run('logistic', config)
            box.check_outputs(self, 'out/v1', 'out/v1.md')

    def test_v2_runner_and_retrospective_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            box.evaluation('eval_retro.json', RETROSPECTIVE, [42])
            primary = box.config('v2.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                 output_directory='out/v2', report='out/v2.md', **LOGISTIC_V2)
            box.run('logistic', primary)
            box.check_outputs(self, 'out/v2', 'out/v2.md')
            # The sensitivity runner itself checks that A, B and A_B (no S) equal the primary run.
            retro = box.config('v2_retro.json', manifest=RETROSPECTIVE, evaluation_config='config/eval_retro.json',
                               output_directory='out/v2_retro', report='out/v2_retro.md',
                               compare_against='out/v2/evaluation/summary.json',
                               sensitivity='retrospective_delay', **LOGISTIC_V2)
            box.run('logistic', retro)
            box.check_outputs(self, 'out/v2_retro', 'out/v2_retro.md')


@unittest.skipUnless(HAS_TORCH, 'needs torch (pip install ".[modeling]")')
class NeuralSmokeTests(unittest.TestCase):
    def test_lstm_v2_runner(self):
        from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl, read_text
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            reference = box.config('reference.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                   output_directory='out/reference', report='out/reference.md', **LOGISTIC_V2)
            box.run('logistic', reference)
            config = box.config('lstm_v2.json', version='v2', mode='full', manifest=PRIMARY,
                                evaluation_config='config/eval.json', output_directory='out/lstm_v2',
                                report='out/lstm_v2.md', compare_against_logistic='out/reference/evaluation/summary.json',
                                seeds=[42], hidden_grid=[4], dropout_grid=[0.], weight_decay_grid=[1e-4],
                                s_weight_decay_grid=[.1], embedding_dim=4, learning_rate=1e-3,
                                batch_size=256, max_epochs=2, patience=2, device='cpu', workers=1)
            box.run('neural', config)
            summary = read_json(box.root / 'out/lstm_v2/summary.json')
            self.assertEqual(set(summary['metrics']), CONDITIONS)
            self.assertEqual(summary['device'], 'cpu')
            self.assertEqual(len(read_jsonl(box.root / 'out/lstm_v2/selection.jsonl')), 98)
            self.assertEqual(len(read_jsonl(box.root / 'out/lstm_v2/epochs.jsonl')), 196)
            self.assertTrue(read_text(box.root / 'out/lstm_v2.md').strip())

    def test_lstm_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            config = box.config('lstm.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                output_directory='out/lstm', report='out/lstm.md', seeds=[42],
                                hidden_dim=4, embedding_dim=4, learning_rate=1e-3, weight_decay=0.0,
                                batch_size=256, max_epochs=2, patience=1)
            box.run('neural', config)
            box.check_outputs(self, 'out/lstm', 'out/lstm.md')

    def test_assayag_reproduction_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            config = box.config('repro.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                output_directory='out/repro', report='out/repro.md', seeds=[42],
                                embedding_dim=4, hidden_dim=4, num_layers=1, dropout_rate=0.0,
                                learning_rate=1e-3, batch_size=256, max_epochs=2, patience=1)
            box.run('neural_reproduction', config)
            self.assertTrue((box.root / 'out/repro.md').read_text(encoding='utf-8').strip())


if __name__ == '__main__':
    unittest.main()
