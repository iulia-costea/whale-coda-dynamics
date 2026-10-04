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


RQ2_D = 'data/processed/rq2_d_rhythm_tempo_folds/manifest.json'
# The RQ2 runs use one BLAS thread: OpenBLAS's default thread count is far slower on these small matrices.
# The own-history length sweep requires one BLAS thread in every fitting process (DEC-25 runs).
ONE_BLAS_THREAD = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


@unittest.skipUnless(HAS_SCIPY, 'needs numpy and scipy (pip install ".[modeling]")')
class Rq2SmokeTests(unittest.TestCase):
    def test_inventory_logistic_runner(self):
        """DEC-22: logistic v4 on an inventory's manifest, with the RQ2 report settings and no comparison run."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy('data/processed/rq2_d_rhythm_tempo_targets.jsonl', RQ2_D)
            box.evaluation('eval.json', RQ2_D, [42])
            config = box.config('rq2.json', manifest=RQ2_D, evaluation_config='config/eval.json',
                                output_directory='out/rq2', report='out/rq2.md', version_label='v4, inventory D',
                                report_title='RQ2 logistic-regression reference results', report_note='Smoke note.',
                                skip_nonconverged_candidates=True, **LOGISTIC_V2)
            box.run('logistic', config, env=ONE_BLAS_THREAD)
            box.check_outputs(self, 'out/rq2', 'out/rq2.md')
            text = (box.root / 'out/rq2.md').read_text(encoding='utf-8')
            self.assertTrue(text.startswith('# RQ2 logistic-regression reference results, v4, inventory D\n'))
            self.assertIn('Smoke note.', text)
            self.assertIn('| Condition | LL bits/coda | Accuracy | Macro-F1 |', text)  # no comparison columns
            self.assertNotIn('Same fourteen leave-one-day-out folds as', text)
            self.assertIn('rests on about 405 non-repeat codas', text)  # 1,547 targets - 1,142 repeats (PLAN [R2-0])
            self.assertIn('reports/rq2_d_rhythm_tempo_folds.md reports UNK rates per fold', text)
            self.assertIn('No candidate failed the convergence check', text)

    def test_compare_runner(self):
        """rq2_compare on the committed config, every inventory standing in as RQ1's logistic v4 run."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy(V4 + 'evaluation/summary.json', V4 + 'evaluation/scored_predictions.jsonl',
                     V4 + 'l2_selection.jsonl', V4 + 'predictions.jsonl')
            values = json.loads((ROOT / 'config' / 'rq2_compare.json').read_text(encoding='utf-8'))
            for inventory in values['inventories']:
                inventory.update(manifest=PRIMARY, results_directory=V4.rstrip('/'))
            values.update(output_directory='out/compare', report='out/compare.md')
            box.run('rq2_compare', box.config('compare.json', **values), env=ONE_BLAS_THREAD)
            summary = json.loads((box.root / 'out/compare/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(len(summary['inventories']), 7)
            for ident, inventory in summary['inventories'].items():
                if ident != summary['reference']:
                    for difference in inventory['share_differences_vs_reference'].values():
                        self.assertEqual(difference['estimate'], 0.0)
            self.assertEqual([p['excluded_targets']['both'] for p in summary['tempo_decomposition']], [31, 31])
            self.assertIn('exploratory', (box.root / 'out/compare.md').read_text(encoding='utf-8'))

    def test_partner_tempo_runner(self):
        """DEC-24 on the committed config with one L2 value and 50 replicates: both runs and the report."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy(*(f'data/processed/{name}' for name in (
                'rq2_c_rhythm_targets.jsonl', 'rq2_c_rhythm_folds/manifest.json', 'rq2_d_rhythm_tempo_targets.jsonl',
                RQ2_D.split('data/processed/')[1], 'rq2_e_type_tempo_targets.jsonl', 'rq2_e_type_tempo_folds/manifest.json')))
            values = json.loads((ROOT / 'config' / 'rq2_partner_tempo.json').read_text(encoding='utf-8'))
            for run in values['runs']:
                run['output_directory'] = 'out/' + run['name']
            values.update(l2_grid=[0.01], bootstrap_replicates=50, report='out/partner_tempo.md')
            box.run('tempo_factor', box.config('partner_tempo.json', **values), env=ONE_BLAS_THREAD)
            for name in ('rhythm', 'type'):
                out = box.root / 'out' / name
                summary = json.loads((out / 'evaluation/summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['conditions'], ['floor', 'own', 'own_T', 'own_P', 'own_S', 'own_T_P', 'own_T_S',
                                                         'own_P_S', 'own_T_P_S'])
                self.assertEqual(summary['comparison_targets'], 1547)
                self.assertEqual(summary['governing_stop_rule']['model'], 'v5')
                for model in ('own_fit', 'v5', 'v5_one_se'):
                    self.assertEqual(len(summary['models'][model]['stop_rules']['new']['checks']), 13)
                    self.assertEqual(set(summary['models'][model]['contrasts']),
                                     {'partner_tempo', 'partner_pattern', 'tempo_beyond_pattern', 'pattern_beyond_tempo'})
                self.assertGreaterEqual(summary['gates']['validation_monotonicity']['min_add_block_validation_contrast_bits'], 0.0)
                self.assertEqual(summary['turn_category_counts'], {'overlap': 441, 'self_continuation': 461, 'switch': 645})
                lines = (out / 'selection.jsonl').read_text(encoding='utf-8').splitlines()
                self.assertEqual(len(lines), 14 * 8 * 2)
                self.assertEqual(len((out / 'models.jsonl').read_text(encoding='utf-8').splitlines()), 14 * 9)
                self.assertIn('data/processed/rq2_c_rhythm_targets.jsonl', summary['input_sha256'])
            text = (box.root / 'out/partner_tempo.md').read_text(encoding='utf-8')
            self.assertTrue(text.startswith('# RQ2 partner-tempo ablation (DEC-24)\n'))
            self.assertIn('| partner type alone | own_P_S − own_S | v5 (max) |', text)
            self.assertIn('By turn category (v5, descriptive)', text)


V4_RUN_FILES = ('models.jsonl', 'l2_selection.jsonl', 'predictions.jsonl', 'baseline_predictions.jsonl',
                'evaluation/summary.json', 'evaluation/scored_predictions.jsonl')


@unittest.skipUnless(HAS_SCIPY, 'needs numpy and scipy (pip install ".[modeling]")')
class LogisticV5SmokeTests(unittest.TestCase):
    def test_v5_runner_retrospective_run_and_compare(self):
        """DEC-23 on the committed RQ1 v4 runs with the committed v5 configs (no fits: v5 only reads v4's outputs),
        then rq2_compare in v5 mode with every inventory standing in as RQ1's v5 run."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy('config/rq1_logistic_v4.json', 'config/rq1_evaluation_v4.json', 'config/rq1_logistic_v5.json',
                     'config/rq1_logistic_v4_retrospective.json', 'config/rq1_evaluation_v4_retrospective.json',
                     'config/rq1_logistic_v5_retrospective.json',
                     *(f'data/results/{run}/{name}' for run in ('rq1_logistic_v4', 'rq1_logistic_v4_retrospective')
                       for name in V4_RUN_FILES))
            box.run('logistic_v5', box.root / 'config/rq1_logistic_v5.json', env=ONE_BLAS_THREAD)
            out = box.root / 'data/results/rq1_logistic_v5'
            summary = box.check_outputs(self, 'data/results/rq1_logistic_v5', 'reports/rq1_logistic_v5.md')
            self.assertFalse((out / 'predictions.jsonl').exists())  # rebuilt from selection.jsonl, never written
            selection = [json.loads(line) for line in (out / 'selection.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(selection), 14 * 8 * 2)
            self.assertEqual(summary['gates']['reproduction']['fits'], 14 * 8)
            self.assertLessEqual(summary['gates']['reproduction']['max_abs_validation_difference_bits'], 1e-9)
            self.assertGreaterEqual(summary['gates']['validation_monotonicity']['min_add_input_validation_contrast_bits'], 0.0)
            self.assertEqual(summary['governing_stop_rule']['model'], 'v4')
            self.assertEqual({len(summary['stop_rules'][m]['new']['checks']) for m in ('v4', 'v5', 'v5_one_se')}, {16})
            # Condition none has one candidate, the prior, so v5's none is v4's none exactly.
            v4 = json.loads((box.root / V4 / 'evaluation/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['metrics']['none'], v4['metrics']['none'])
            text = (box.root / 'reports/rq1_logistic_v5.md').read_text(encoding='utf-8')
            self.assertTrue(text.startswith("# RQ1 logistic regression v5 (DEC-23)\n\n**Logistic v4 remains RQ1's headline"))
            self.assertIn('Two-sided sign-flip p', text)
            # The old-delay run checks that its conditions without S equal the primary v5 run's.
            box.run('logistic_v5', box.root / 'config/rq1_logistic_v5_retrospective.json', env=ONE_BLAS_THREAD)
            self.assertIn('Primary timing vs the old delay, v5',
                          (box.root / 'reports/rq1_retrospective_sensitivity_v5.md').read_text(encoding='utf-8'))
            values = json.loads((ROOT / 'config' / 'rq2_compare_v5.json').read_text(encoding='utf-8'))
            for inventory in values['inventories']:
                inventory.update(manifest=PRIMARY, results_directory='data/results/rq1_logistic_v5',
                                 v4_results_directory=V4.rstrip('/'))
            values.update(output_directory='out/compare_v5', report='out/compare_v5.md')
            box.run('rq2_compare', box.config('compare_v5.json', **values), env=ONE_BLAS_THREAD)
            compared = json.loads((box.root / 'out/compare_v5/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(compared['model'], 'logistic v5 (DEC-23)')
            for ident, inventory in compared['inventories'].items():
                if ident != compared['reference']:
                    for difference in inventory['share_differences_vs_reference'].values():
                        self.assertEqual(difference['estimate'], 0.0)
            self.assertEqual([p['excluded_targets']['both'] for p in compared['tempo_decomposition']], [31, 31, 31])
            report = (box.root / 'out/compare_v5.md').read_text(encoding='utf-8')
            self.assertIn('logistic v5 (DEC-22, DEC-23)', report)
            self.assertIn('tempo-boundary sensitivity', report)


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
            self.assertEqual({k: len(v['checks']) for k, v in summary['stop_rules_dec23'].items()}, {'1': 16, '3': 16, '4': 16})
            self.assertIn('sign_flip_p_two_sided', summary['contrasts']['A_B_S']['4']['components']['log_likelihood_bits'])
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


@unittest.skipUnless(HAS_TORCH, 'needs torch (pip install ".[modeling]")')
class SwitchStructureSmokeTests(unittest.TestCase):
    def test_switch_structure_runner(self):
        """DEC-28: describe, fit (variants R, P, PD, all conditions), report; then the report alone regenerates byte for byte."""
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy(V4 + 'predictions.jsonl', V4 + 'l2_selection.jsonl', V4 + 'evaluation/summary.json')
            box.evaluation('eval.json', PRIMARY, [42])
            box.config('logistic_tiny.json', **LOGISTIC_V2)
            config = box.config(
                'switch.json', manifest=PRIMARY, logistic_config='config/logistic_tiny.json', evaluation_config='config/eval.json',
                output_directory='out/switch', report='out/switch.md',
                variants=dict(R=[], P=['run_length', 'repertoire'], PD=['run_length', 'repertoire', 'distinct_1', 'distinct_2']),
                reference_variant='R', conditions=['A', 'A_B_S'], primary=dict(condition='A_B_S', full='PD', reduced='P'),
                secondary=dict(condition='A_B_S', full='PD', reduced='R'), descriptive_contrasts=[['P', 'R']],
                repertoire_contrast=dict(condition='A_B_S', full='P', reduced='R'),
                gate=dict(enforce=False, results_directory=V4.rstrip('/'), maximum_penalty_differences=1,
                          log_likelihood_tolerance_bits=1e-5, probability_listing_threshold=1e-5),
                descriptive=dict(enforce=True, bootstrap_replicates=50, bootstrap_seed=42, confidence=0.95,
                                 interval_tolerance=0.015, source='smoke test',
                                 expected={'targets': {'count': 1547}, 'switches_with_d2': {'count': 58}}))
            box.run('switch_structure', config, '--workers', '3', env=ONE_BLAS_THREAD)
            summary = json.loads((box.root / 'out/switch/summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['descriptive']['passed'])
            self.assertEqual(set(summary['metrics']), {'R', 'P', 'PD'})
            self.assertEqual({c: list(v) for c, v in summary['contrasts'].items()},
                             {'A_B_S': ['PD-P', 'PD-R', 'P-R'], 'A': ['PD-P', 'PD-R', 'P-R']})
            self.assertEqual(summary['contrasts']['A_B_S']['PD-P']['role'], 'primary')
            self.assertEqual(summary['contrasts']['A_B_S']['PD-R']['role'], 'secondary')
            self.assertEqual(summary['contrasts']['A']['PD-P']['role'], 'descriptive')
            self.assertIn('sign_flip_p_two_sided', summary['contrasts']['A_B_S']['PD-P']['components']['log_likelihood_bits'])
            self.assertEqual({v: len(r['checks']) for v, r in summary['stop_rules_dec23'].items()}, {'R': 16, 'P': 16, 'PD': 16})
            self.assertEqual(set(summary['coefficients']['PD']['A']), {'copy_beta', 'run_length', 'repertoire', 'distinct_1', 'distinct_2'})
            self.assertEqual(len(summary['coefficients']['PD']['A_B_S']['distinct_1']['per_fold']), 14)
            self.assertEqual(summary['constant_conditions'], ['none', 'S', 'B', 'B_S'])
            self.assertEqual(summary['gate']['fits_compared'], 98)
            self.assertTrue((box.root / 'out/switch.md').read_text(encoding='utf-8').startswith('# RQ1 switch structure'))
            regenerated = ('out/switch.md', 'out/switch/summary.json')
            before = {rel: (box.root / rel).read_bytes() for rel in regenerated}
            box.run('switch_structure', config, '--stage', 'report')
            self.assertEqual({rel: (box.root / rel).read_bytes() for rel in regenerated}, before)


RQ3_INPUTS = ('data/processed/rq3/rq3_codas.csv','data/processed/rq3/rq3_folds.csv',
              'data/results/rq3/calibration/self_scores.csv', 'data/results/rq3/calibration/self_calibration.csv',
              'data/results/rq3/transfer_matrix/scores.csv', 'data/results/rq3/transfer_matrix/summary.csv')


@unittest.skipUnless(HAS_TORCH, 'needs torch (pip install ".[modeling]")')
class Rq3NeuralSmokeTests(unittest.TestCase):
    def test_pooled_runner_all_stages(self):
        """Every [R3-8] stage on the full RQ3 data with one-epoch, hidden-4 models; the gate fails, so the
        rerun path runs too (a smoke config records the gate instead of stopping)."""
        config = json.loads((ROOT / 'config/rq3_neural.json').read_text(encoding='utf-8'))
        config['model'].update(hidden_grid=[4], dropout_grid=[0.0], max_epochs=1, patience=1, refit_seeds=[0])
        config['gate_rerun'].update(hidden_grid=[4, 8], patience=1)
        config['arm5'].update(shares=[0.5, 1.0], draws=1, reduced_shares=[1.0], reduced_draws=1)
        config['pilot'].update(cpu_workers=[2], gpu_workers=[], timing_epochs=1)
        config.update(pooled_alphas=[1.0, 0.5], bootstrap_draws=20, device='cpu', workers=2,
                      require_clean_tree=False, smoke_test=True)
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            box.copy(*RQ3_INPUTS)
            path = box.config('rq3_neural_smoke.json', **config)
            box.run('rq3_neural.runner', path, '--stage', 'all', env={'CUDA_VISIBLE_DEVICES': ''})
            out = box.root / 'data/results/rq3_neural'
            report = (box.root / 'reports/rq3_neural.md').read_text(encoding='utf-8')
            self.assertTrue(report.startswith('# RQ3 pooled multi-clan LSTM'))
            self.assertIn('SMOKE TEST', report)
            summary = json.loads((out / 'report/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(len(summary['predictions']), 5)
            self.assertFalse(summary['gate1_passed']['arm1'])
            self.assertIn('arm1_rerun', summary['gate1_passed'])
            for stage in ('pilot', 'arm1', 'arm1_rerun', 'arm23', 'capacity', 'arm4', 'arm5'):
                stage_summary = json.loads((out / stage / 'summary.json').read_text(encoding='utf-8'))
                self.assertIn('data/processed/rq3/rq3_codas.csv', stage_summary['input_sha256'])
            pairs = (out / 'arm23/tag_pairs.csv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(pairs), 1 + 2 * 6 + sum(1 for line in pairs[1:] if line.startswith('refit')))
            self.assertTrue(all(',True,' in line for line in pairs[1:]))
            before = (box.root / 'reports/rq3_neural.md').read_bytes()
            box.run('rq3_neural.runner', path, '--stage', 'report', env={'CUDA_VISIBLE_DEVICES': ''})
            self.assertEqual((box.root / 'reports/rq3_neural.md').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
