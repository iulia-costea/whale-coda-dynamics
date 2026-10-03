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

    def copy(self, *relative_paths):
        for rel in relative_paths:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, self.root / rel)

    def config(self, name, **values):
        path = self.root / 'config' / name
        path.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
        return path

    def evaluation(self, name, manifest, seeds):
        return self.config(name, manifest=manifest, predictions='unused', output_directory='unused',
                           seeds=seeds, bootstrap_replicates=50, bootstrap_seed=42, confidence=0.95)

    def run(self, module, config, *args, env=None):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), **(env or {}))
        result = subprocess.run([sys.executable, '-m', f'whale_coda_dynamics.{module}', '--config', str(config), *args],
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

    def test_lstm_v2_full_runner(self):
        """DEC-17 stage 2: full manifest, one value per grid axis, 2 epochs, one seed."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            config = box.config('lstm_v2.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                output_directory='out/lstm_v2', report='out/lstm_v2.md', seeds=[42],
                                embedding_dim=4, learning_rate=1e-3, batch_size=256, max_epochs=2, patience=1,
                                dropout_grid=[0.0], hidden_grid=[4], weight_decay_history_grid=[1e-4],
                                weight_decay_s_grid=[1e-4])
            box.run('neural', config)
            box.check_outputs(self, 'out/lstm_v2', 'out/lstm_v2.md')

    def test_lstm_v2_pilot_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.evaluation('eval.json', PRIMARY, [42])
            config = box.config('lstm_v2_pilot.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                                output_directory='out/lstm_v2_pilot', report='out/lstm_v2_pilot.md', seeds=[42],
                                test_days=['2014-04-16', '2015-04-01'], conditions=['A'],
                                embedding_dim=4, learning_rate=1e-3, batch_size=256, max_epochs=2, patience=1,
                                dropout_grid=[0.0], hidden_grid=[4], weight_decay_history_grid=[1e-4],
                                weight_decay_s_grid=[1e-4])
            box.run('neural', config)
            box.check_outputs(self, 'out/lstm_v2_pilot', 'out/lstm_v2_pilot.md', conditions={'A'})

    def test_lstm_v2_suppression_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            config = box.config('suppression.json', manifest=PRIMARY, report='out/suppression.md',
                                test_day='2014-04-16', fields_without_s=['A_history'],
                                fields_with_s=['A_history', 'turn_category', 'gap_s', 'overlap_elapsed_s'],
                                seeds=[42], device='cpu', embedding_dim=4, hidden_dim=4, dropout_rate=0.0,
                                weight_decay_history=1e-4, weight_decay_s=1.0, learning_rate=1e-3,
                                batch_size=256, max_epochs=2, patience=2, tolerance_bits=0.01)
            box.run('neural', config)
            self.assertTrue((box.root / 'out/suppression.md').read_text(encoding='utf-8').strip())

    def test_lstm_v2_timing_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            config = box.config('timing.json', manifest=PRIMARY, report='out/timing.md',
                                test_day='2014-04-16', fields=['A_history'], seeds=[42, 43],
                                devices=['cpu', 'cuda'], embedding_dim=4, hidden_dim=4, dropout_rate=0.0,
                                weight_decay_history=1e-4, weight_decay_s=None, learning_rate=1e-3,
                                batch_size=256, max_epochs=2, patience=2, full_study_seeds=[42, 43, 44],
                                full_study_folds=14, dropout_grid=[0.0], hidden_grid=[4],
                                weight_decay_history_grid=[1e-4], weight_decay_s_grid=[1e-4])
            box.run('neural', config)
            text = (box.root / 'out/timing.md').read_text(encoding='utf-8')
            self.assertIn('cpu', text)
            self.assertIn('not available', text)  # no GPU on the CI/dev machine

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


V4 = 'data/results/rq1_logistic_v4/'
V4_INPUTS = dict(logistic_models=V4 + 'models.jsonl', logistic_predictions=V4 + 'predictions.jsonl')
TINY_RESIDUAL = dict(seeds=[42], embedding_dim=4, learning_rate=1e-3, weight_decay=0.1, batch_size=256, max_epochs=2,
                     patience=1, hidden_grid=[4], dropout_grid=[0.0])


@unittest.skipUnless(HAS_TORCH, 'needs torch (pip install ".[modeling]")')
class ResidualSmokeTests(unittest.TestCase):
    def test_residual_checks_throughput_and_full_runner(self):
        """DEC-21: the checks, the device measurement (CPU only here) and the full run that reads both."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy('config/rq1_logistic_v4.json', V4 + 'models.jsonl', V4 + 'predictions.jsonl', V4 + 'l2_selection.jsonl',
                     V4 + 'evaluation/scored_predictions.jsonl', V4 + 'evaluation/summary.json')
            checks = box.config('checks.json', manifest=PRIMARY, logistic_config='config/rq1_logistic_v4.json',
                                logistic_selection=V4 + 'l2_selection.jsonl', output_directory='out/checks',
                                report='out/checks.md', workers=2, offset_tolerance=1e-9, **V4_INPUTS, **TINY_RESIDUAL,
                                checks=dict(wiring=dict(hidden=4, dropout=0.0, seed=42, epoch0_tolerance_bits=1e-12),
                                            learnability=dict(test_days=['2015-03-02'], conditions=['A'], hidden=4,
                                                              dropout=0.0, max_epochs=2, seeds=[42], minimum_gain_bits=-1e9),
                                            planted=dict(test_day='2015-03-02', lag=6, fractions=[0.0, 0.4], sampling_seed=42)))
            box.run('residual', checks)
            summary = json.loads((box.root / 'out/checks/summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['gates']['wiring'])
            throughput = box.config('throughput.json', manifest=PRIMARY, logistic_models=V4 + 'models.jsonl',
                                    output_directory='out/throughput', report='out/throughput.md', **TINY_RESIDUAL,
                                    throughput=dict(test_day='2015-03-02', condition='A_B_S', hidden=4, dropout=0.0,
                                                    seeds=[42], cpu_workers=2, gpu_processes=[], tie_tolerance=0.05))
            box.run('residual', throughput)
            box.evaluation('eval.json', PRIMARY, [42])
            full = box.config('residual.json', manifest=PRIMARY, evaluation_config='config/eval.json',
                              checks_summary='out/checks/summary.json', throughput_summary='out/throughput/summary.json',
                              logistic_scored=V4 + 'evaluation/scored_predictions.jsonl',
                              logistic_summary=V4 + 'evaluation/summary.json', output_directory='out/residual',
                              report='out/residual.md', offset_tolerance=1e-9, enforce_check_gates=False,
                              **V4_INPUTS, **TINY_RESIDUAL)
            box.run('residual', full)
            summary = box.check_outputs(self, 'out/residual', 'out/residual.md')
            # Condition none is logistic v4's prior on every fold: its gain must be exactly zero, not float noise.
            none = summary['residual_gain']['none']
            self.assertEqual((none['zero_days'], none['positive_days'], none['mean_bits_per_coda']), (14, 0, 0.0))
            # Regenerating from the committed outputs, without refitting, reproduces the full run byte for byte.
            regenerated = ('out/residual.md', 'out/residual/evaluation/summary.json',
                           'out/residual/evaluation/scored_predictions.jsonl', 'out/residual/evaluation/paired_contrasts.jsonl')
            before = {rel: (box.root / rel).read_bytes() for rel in regenerated}
            box.run('residual', full, '--report-only')
            self.assertEqual({rel: (box.root / rel).read_bytes() for rel in regenerated}, before)
            # Planted-signal sensitivity run for condition A on every fold (synthetic labels), against the run above.
            box.config('logistic_tiny.json', history_order=3, l2_grid=[0.01], max_iterations=2000, gradient_tolerance=1e-6)
            planted = box.config('planted.json', manifest=PRIMARY, logistic_config='config/logistic_tiny.json',
                                 real_run_summary='out/residual/evaluation/summary.json',
                                 real_run_scored='out/residual/evaluation/scored_predictions.jsonl',
                                 throughput_summary='out/throughput/summary.json', output_directory='out/planted',
                                 report='out/planted.md', fractions=[0.1, 0.4], lag=6, sampling_seed=42,
                                 bootstrap_replicates=50, bootstrap_seed=42, confidence=0.95, ratio_handful=5,
                                 **TINY_RESIDUAL)
            box.run('residual_planted', planted)
            summary = json.loads((box.root / 'out/planted/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(set(summary['per_fraction']), {'0.1', '0.4'})
            self.assertEqual(len(summary['counts']), 14)
            self.assertTrue((box.root / 'out/planted.md').read_text(encoding='utf-8').strip())


# The own-history length sweep requires one BLAS thread in every fitting process (DEC-25 runs).
ONE_BLAS_THREAD = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
PLANTED_SCORES = 'data/results/rq1_lstm_residual_planted/scores.jsonl'


@unittest.skipUnless(HAS_TORCH, 'needs torch (pip install ".[modeling]")')
class HistoryLengthSmokeTests(unittest.TestCase):
    def test_sweep_runner(self):
        """DEC-25: fit (k = 1, 3, 4, all conditions), calibrate, report; then the report alone regenerates byte for byte."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy(V4 + 'predictions.jsonl', V4 + 'l2_selection.jsonl', V4 + 'evaluation/summary.json', PLANTED_SCORES)
            box.evaluation('eval.json', PRIMARY, [42])
            box.config('logistic_tiny.json', **LOGISTIC_V2)
            box.config('planted_tiny.json', fractions=[0.4], lag=6, sampling_seed=42, bootstrap_replicates=50,
                       bootstrap_seed=42, confidence=0.95, ratio_handful=5)
            config = box.config('sweep.json', manifest=PRIMARY, logistic_config='config/logistic_tiny.json',
                                evaluation_config='config/eval.json', planted_config='config/planted_tiny.json',
                                output_directory='out/sweep', report='out/sweep.md', field='A_history', orders=[1, 3, 4],
                                reference_order=3, conditions=['A', 'A_B_S'], primary=dict(condition='A_B_S', order=4),
                                secondary_order=1,
                                gate=dict(enforce=False, results_directory=V4.rstrip('/'), maximum_penalty_differences=1,
                                          log_likelihood_tolerance_bits=1e-5, probability_listing_threshold=1e-5),
                                calibration=dict(condition='A', order=4, residual_planted_scores=PLANTED_SCORES))
            box.run('history_length', config, '--workers', '2', env=ONE_BLAS_THREAD)
            summary = json.loads((box.root / 'out/sweep/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(set(summary['curve']['A']), {'1', '3', '4'})
            self.assertEqual({c: set(v) for c, v in summary['contrasts'].items()}, {'A': {'1', '4'}, 'A_B_S': {'1', '4'}})
            self.assertEqual(summary['contrasts']['A_B_S']['4']['role'], 'primary')
            self.assertEqual(summary['contrasts']['A']['1']['role'], 'secondary')
            self.assertEqual(set(summary['stop_rules']), {'1', '3', '4'})
            self.assertEqual(summary['gate']['fits_compared'], 98)
            self.assertFalse(summary['gate']['passed'])  # a one-value L2 grid cannot reproduce v4's choices
            self.assertEqual(set(summary['calibration']['per_fraction']), {'0.4'})
            self.assertTrue(summary['calibration']['planting_check']['flags_and_labels_identical'])
            self.assertEqual(summary['constant_conditions'], ['none', 'S', 'B', 'B_S'])
            self.assertTrue((box.root / 'out/sweep.md').read_text(encoding='utf-8').startswith('# RQ1 own-history length sweep'))
            regenerated = ('out/sweep.md', 'out/sweep/summary.json')
            before = {rel: (box.root / rel).read_bytes() for rel in regenerated}
            box.run('history_length', config, '--stage', 'report')
            self.assertEqual({rel: (box.root / rel).read_bytes() for rel in regenerated}, before)


if __name__ == '__main__':
    unittest.main()
