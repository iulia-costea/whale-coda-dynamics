# Approved RQ1 execution prompt (2026-09-29)

The user approved executing the original prompt with the following amendments. These amendments take precedence over the original text below.

1. Preserve logistic v3, commit and push it, and open a draft PR even if its DEC-16 likelihood stop rule fires. Report the failure explicitly. This exception does not waive convergence, provenance or test failures.
2. Record the separate-penalty decision as DEC-20: DEC-19 already describes RQ3 and must remain intact. History blocks A/B and their copy weights receive one validation-selected penalty; S receives another. Conditions without S are unchanged. This permits independent shrinkage in support of optional ignorance; it does not guarantee better held-out performance. The user reports that S reduced best validation likelihood on 10/14 v2 folds; verify this from artifacts before presenting it as a reproduced finding.
3. Logistic v4 jointly searches history L2 and S L2 over [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]. Ties prefer larger history L2, then larger S L2. The shared intercept retains the history/base penalty. An S-only condition therefore still selects its intercept/base and S penalties jointly. No-S conditions retain v3 exactly. Record both boundary counts. Rerun gate 1 and the old-delay sensitivity; v4 becomes the reference for stages 2 and 3. Stop on any v4 gate failure.
4. LSTM v2 jointly selects history weight decay [1e-5, 1e-4, 1e-3], S weight decay [1e-5, 1e-3, 0.1, 1.0, 10.0], dropout [0, 0.2, 0.4], and hidden size [32, 64]. History encoders, copy weights, their output columns and shared output bias use history/base decay; the S encoder and its output columns use S decay. Absent blocks add no parameters. Ties prefer smaller hidden size, larger dropout, larger history decay, then larger S decay. No-S conditions retain the original search. Pilot timing must report the expanded search cost.
5. Before the pilot, use one predeclared fold's validation day to check that the strongest S penalty makes A_B_S validation LL close to A_B (absolute difference <= 0.01 bits/coda, mean across seeds 42/43/44, matching remaining settings). Treat failure as a diagnostic stop, not permission to retune on test results. Define CPU/GPU seed spread as max minus min validation LL across the three seeds; compare the absolute difference in device means against each device's spread.
6. The original stop rules remain unchanged apart from the explicit v3 preservation exception. Stage 3 requires CUDA availability; this macOS machine can perform CPU work through stage 2 only.
7. Implementation is approved; do not request plan approval again. Full baseline tests must pass before model implementation. Commit code before generating research results, commit generated results, and verify a clean checkout reproduces them. Keep old results untouched.

## Original prompt, subject to the amendments above

Read AGENTS.md and CLAUDE.md, then docs/PLAN.md §4 ([M1b], [M3], [M4]) and §5, docs/DECISIONS.md DEC-16 and DEC-17, and reports/README.md. For what went wrong before, read reports/rq1_lstm.md (LSTM v1, superseded). Code: src/whale_coda_dynamics/neural.py, logistic.py, evaluation.py.

Context: LSTM v1 used fixed WhaleLM settings and had no copy path; it fell below logistic regression v2 on 6 of 7 conditions. DEC-16/17 fix this. Do not change preprocessing, folds, the UNK policy, the S definition or the evaluation logic.

Work in three stages on one branch. Each stage ends with a gate. If every check in it passes, commit, push and continue without waiting for me. If any check fails, stop and report to me; do not change settings, grids or rules to make a check pass.

STAGE 1 — logistic regression v3 (DEC-16) and report hygiene
1. New config/output/report names for v3 (keep v1 and v2 untouched): L2 grid [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]; everything else as v2.
2. Stop rules: replace the accuracy-based stop conditions with DEC-16's log-likelihood-only rules, in logistic.py and neural.py (shared helper).
3. Report writers: round CI bounds to 4 decimals in the markdown reports (JSON keeps full precision).
4. Run v3, then the old-delay sensitivity run on v3 (same machine: it must match the primary run exactly for conditions without S).
5. Produce v3 from main's committed data (clean checkout). tests/test_committed_artifacts.py lists older results fitted before the CRLF provenance fix (CRLF_PROVENANCE); v3's summaries must match the committed files exactly, with no new entries. Leave v1/v2 and their entries untouched.
Stage summary (post it as a comment on the draft PR): chosen L2 per condition (distribution over folds; how many at the new lower edge 0.0001), the three main contrasts and all twelve add-one-input contrasts v2 vs v3 (token-weighted and unweighted means, CI, positive days), and every A-condition vs repeat-own-last.
GATE 1 (all must hold):
- the full test suite passes;
- every v3 fit converged;
- no DEC-16 stop rule fires;
- for no condition is the chosen L2 at the lower edge 0.0001 in the majority of folds (that would mean the grid is still too narrow);
- tests/test_committed_artifacts.py passes with the v3 results committed.
If all hold: commit code and results, push, open a draft PR to main (ask me to open it if gh isn't available), post the stage summary, and continue with stage 2.

STAGE 2 — LSTM v2 pilot (DEC-17)
1. Implement in neural.py, with a new config and output directory (rq1_lstm_v2); keep v1 files untouched:
   - Copy weights: a learnable scalar beta added to the output logit of the class equal to the whale's own last coda (last entry of the mapped A_history), only in conditions with A; gamma likewise for the partner's last coda (mapped B_history), only in conditions with B. Absent input → no weight.
   - Adam, learning rate 1e-3, batch 32, up to 300 epochs, patience 20, restore the best-validation epoch.
   - Dropout on the encoder outputs; weight decay; hidden size. Selection per fold and condition on the validation day from dropout {0, 0.2, 0.4} × weight decay {1e-5, 1e-4, 1e-3} × hidden {32, 64}, by validation log-likelihood averaged over seeds; ties go to the smaller, more regularised setting. Identical grid and rule for every condition.
   - Log training and validation log-likelihood per epoch for every fit (jsonl).
   - Device: the code currently never uses the GPU. Add a config setting "device" ("cpu" or "cuda", default "cpu"), identical for every condition, recorded in the summary. Tests and the smoke test always run on CPU.
2. Tests: gradient reaches beta/gamma; with the LSTM part zeroed and a large beta the argmax repeats the own last coda on every row; conditions without A/B have no beta/gamma; selection never reads test rows (concretely, as test_selection_uses_validation_only in tests/test_logistic.py: changing test-day labels or inputs leaves the chosen setting unchanged). Add a smoke test for the LSTM v2 runner to tests/test_pipeline_smoke.py (full manifest, one value per grid axis, 2 epochs, one seed).
3. Pilot: conditions A and A_B_S on test days 2015-03-02, 2015-03-31, 2015-03-26 (largest) and 2016-04-24 (small), seed 42, full grid.
Stage summary (PR comment), per fold and condition: LSTM v2 vs logistic v3 vs repeat-own-last (LL bits/coda, accuracy, macro-F1), chosen setting, fitted beta/gamma, epoch of best validation LL, and whether any run hit 300 epochs. Include a small learning-curve summary (train vs validation LL) for one fold. Timing and determinism: for one fold and condition, fit the chosen setting with seeds 42, 43 and 44 on CPU, and, if you have a CUDA GPU, on GPU. Report seconds per fit on each device, and estimate stage 3's wall time on CPU (parallel fits across cores) and on GPU. GPU only: with torch.use_deterministic_algorithms(True) and the environment variables that PyTorch's nn.LSTM documentation gives for deterministic cuDNN RNNs, a second GPU run with the same seed must reproduce the first exactly. CPU and GPU runs will not match exactly (different kernels, and small differences grow over training); instead, the CPU–GPU difference in mean validation log-likelihood must be smaller than the spread across the three seeds.
GATE 2 (all must hold):
- the full test suite passes, including the new tests and the LSTM v2 smoke test;
- for A and A_B_S, LSTM v2's mean log-likelihood over the 4 pilot folds is not below repeat-own-last (CLAUDE.md: below it = fitting failure);
- for neither condition is LSTM v2 below logistic v3 on all 4 pilot folds (4-fold version of DEC-17's [U1] rule);
- no fit with the chosen setting hit the 300-epoch cap;
- GPU only: both determinism checks above pass.
If a check fails: stop and report. Do not retune on the pilot results: they are test days.
If all hold: commit, push and post the stage summary. If a CUDA GPU is available, continue with stage 3; otherwise stop here and say so.

STAGE 3 — full LSTM v2 (only if gate 2 passed and a CUDA GPU is available)
1. All 8 conditions × 14 folds × 3 seeds × full grid; the chosen setting's 3 seeds are the reported runs. Device: whichever the pilot showed is faster (both passed the determinism checks); state it in the report.
2. Report as reports/rq1_lstm_v2.md with the same tables as v1, plus: LSTM v2 vs logistic v3 per condition with paired CIs ([U1]); distribution of chosen settings; the Assayag context check without retraining (condition B on switch/overlap targets only vs the majority class on that same subset; condition A vs Assayag et al. 2026 Table 3 single-whale LSTM accuracies 0.746–0.922, noting their split is random and ours holds out days).
3. Stop rules per DEC-16/17 (log-likelihood only); if any fires, stop and show me before writing conclusions.
4. Update reports/README.md: v2/v3 statuses.
5. Commit, push, post the stage summary and mark the PR ready for review.

Rules: before stage 1, propose a plan for all three stages (files, interfaces, anything ambiguous or in conflict with PLAN) and do not edit until I approve it; after that, work through the stages without waiting for me, stopping only where a gate fails or tells you to. The full test suite (`PYTHONPATH=src python -m pytest tests`, see CLAUDE.md) passes before you start and at every gate; never weaken or skip a test, and never change a setting, grid or rule to make a gate pass: stop and report instead. All text I/O through io_utils.py. Commit code before producing results, then rerun from a clean checkout and check that git status is clean (CLAUDE.md).

