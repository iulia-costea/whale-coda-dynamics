# RQ1 residual LSTM: planted-signal sensitivity, condition A

Synthetic labels only; no result on real data changes. Protocol adapted from Bastings et al. (2022), "'Will You Find These Shortcuts?' A Protocol for Evaluating the Faithfulness of Input Salience Methods for Text Classification" (EMNLP): partially synthetic data with a known planted rule. They add examples; here existing targets are relabelled.

On all 14 folds, a fraction f in [0.1, 0.2, 0.4] of the eligible targets on training, validation and test days is relabelled as the whale's own coda at lag 6. Eligible: at least 6 own codas, and the coda at lag 6 differs from lags 1-3 (as in check 6c). Labels only; histories are unchanged. Planted subsets are nested across f: one fixed permutation per fold and split (RandomState seeded from SHA-256 of '42:fold:split'). Logistic regression is refitted per fold and f on the planted training and validation days with v4's procedure (L2 chosen on the validation day), and condition A's residual selection runs exactly as in the full run: hidden [16, 32, 64] x dropout [0.0, 0.2, 0.4] plus logistic regression alone, seeds [42, 43, 44, 45, 46], the one-SE rule. Device: cpu with 18 processes.

Scored on the planted test labels, over all test targets of the 14 days. (1) residual network minus the refitted logistic regression; (2) an oracle minus the refitted logistic regression, where the oracle predicts f x 1[own coda at the lag] + (1 - f) x P_LR on eligible test targets and P_LR elsewhere (it knows the rule and f, not which targets were planted); (3) the recovery ratio r = (1)/(2). Log-likelihoods use evaluation.log_likelihood_bits; intervals are evaluation.day_bootstrap_bits (10000 replicates). r is a ratio of means: in each replicate, (1)'s token-weighted mean over the resampled days divided by (2)'s over the same days. The oracle loses log2(1 - f) on eligible targets that were not planted, so a replicate can have an oracle difference at or below zero; those replicates have no ratio and are counted (column "Non-positive"); r's interval is over the rest. More than 5 such replicates make r unstable at that f. Per-coda residual values are means over seeds of per-seed differences.

## Detection

| f | Planted test targets (share) | Folds with A selected | (1) Residual - LR | CI | Positive days | (2) Oracle - LR | CI | (3) r | r CI | Non-positive |
|---:|---|---:|---:|---|---:|---:|---|---:|---|---:|
| 0.1 | 11/1547 (0.7%) | 8/14 | -0.00768 | [-0.0397, 0.0165] | 5/14 | 0.03513 | [0.0185, 0.0550] | -0.2187 | [-1.166, 0.5025] | 0/10000 |
| 0.2 | 23/1547 (1.5%) | 11/14 | -0.01649 | [-0.0544, 0.0230] | 6/14 | 0.06903 | [0.0479, 0.0896] | -0.2389 | [-0.7916, 0.3794] | 0/10000 |
| 0.4 | 43/1547 (2.8%) | 11/14 | 0.00575 | [-0.0498, 0.0549] | 6/14 | 0.13702 | [0.0960, 0.1808] | 0.04199 | [-0.3605, 0.4565] | 0/10000 |

## Bound

The real-data residual gain of condition A over logistic v4 has a day-bootstrap upper bound of 0.0071 bits/coda (reports/rq1_lstm_residual.md). Where r's lower bound is above zero, that upper bound divided by r's lower bound is the oracle gain above which a pattern of this form (a deterministic copy of the own coda at lag 6, which logistic regression's three lags cannot express) would have pushed the residual gain past the observed upper bound. It is stated only where the oracle difference stays positive in every resample.

- f = 0.1: r's lower bound is -1.166, at or below zero, so the run gives no bound.
- f = 0.2: r's lower bound is -0.7916, at or below zero, so the run gives no bound.
- f = 0.4: r's lower bound is -0.3605, at or below zero, so the run gives no bound.

## Bastings et al.'s checks

(a) Targets left unplanted: the residual network trained on planted data next to the real-data run on the same targets and labels (planting should not degrade them), with the refitted logistic regression for context. Difference: planted-data residual minus real-data residual, day bootstrap.

| f | Targets | Residual (planted run) | Residual (real-data run) | Difference | CI | Refitted LR |
|---:|---:|---:|---:|---:|---|---:|
| 0.1 | 1536 | -1.11409 | -1.12002 | 0.00593 | [-0.0146, 0.0298] | -1.10792 |
| 0.2 | 1524 | -1.14192 | -1.12253 | -0.01940 | [-0.0620, 0.0182] | -1.12802 |
| 0.4 | 1504 | -1.13440 | -1.12221 | -0.01220 | [-0.0490, 0.0241] | -1.13078 |

(b) Planted test targets: mean log2-probability of the planted label.

| f | Targets | Residual network | Refitted LR | Oracle |
|---:|---:|---:|---:|---:|
| 0.1 | 11 | -8.131 | -7.912 | -3.233 |
| 0.2 | 23 | -7.406 | -7.218 | -2.226 |
| 0.4 | 43 | -6.565 | -6.899 | -1.278 |

## Counts per fold

Eligible and planted targets per split (training / validation / test), and the planted share of each fold's targets over all three splits.

| Fold | Targets | Eligible | Planted, f = 0.1 | Planted, f = 0.2 | Planted, f = 0.4 |
|---|---|---|---|---|---|
| test-2014-04-16 | 1361/126/60 | 93/10/7 | 9/1/1 (0.7%) | 19/2/1 (1.4%) | 37/4/3 (2.8%) |
| test-2015-04-01 | 1337/84/126 | 100/0/10 | 10/0/1 (0.7%) | 20/0/2 (1.4%) | 40/0/4 (2.8%) |
| test-2015-04-10 | 1433/30/84 | 105/5/0 | 11/1/0 (0.8%) | 21/1/0 (1.4%) | 42/2/0 (2.8%) |
| test-2016-05-13 | 1489/28/30 | 105/0/5 | 11/0/1 (0.8%) | 21/0/1 (1.4%) | 42/0/2 (2.8%) |
| test-2014-04-24 | 1488/31/28 | 109/1/0 | 11/0/0 (0.7%) | 22/0/0 (1.4%) | 44/0/0 (2.8%) |
| test-2015-04-07 | 1402/114/31 | 96/13/1 | 10/1/0 (0.7%) | 19/3/0 (1.4%) | 38/5/0 (2.8%) |
| test-2016-05-12 | 1385/48/114 | 97/0/13 | 10/0/1 (0.7%) | 19/0/3 (1.4%) | 39/0/5 (2.8%) |
| test-2014-04-29 | 1433/66/48 | 107/3/0 | 11/0/0 (0.7%) | 21/1/0 (1.4%) | 43/1/0 (2.8%) |
| test-2015-03-19 | 1465/16/66 | 107/0/3 | 11/0/0 (0.7%) | 21/0/1 (1.4%) | 43/0/1 (2.8%) |
| test-2016-04-24 | 1479/52/16 | 106/4/0 | 11/0/0 (0.7%) | 21/1/0 (1.4%) | 42/2/0 (2.8%) |
| test-2016-04-12 | 1141/354/52 | 70/36/4 | 7/4/0 (0.7%) | 14/7/1 (1.4%) | 28/14/2 (2.8%) |
| test-2015-03-02 | 911/282/354 | 61/13/36 | 6/1/4 (0.7%) | 12/3/7 (1.4%) | 24/5/14 (2.8%) |
| test-2015-03-31 | 1009/256/282 | 79/18/13 | 8/2/1 (0.7%) | 16/4/3 (1.5%) | 32/7/5 (2.8%) |
| test-2015-03-26 | 1231/60/256 | 85/7/18 | 9/1/2 (0.8%) | 17/1/4 (1.4%) | 34/3/7 (2.8%) |

Chosen candidate per fold and f, and the selection behind it, are in data/results/rq1_lstm_residual_planted/selection.jsonl; per-target scores in scores.jsonl; every fit in fits.jsonl; summary.json records SHA-256 of every input.

This is a sensitivity check of the model and selection rule on one synthetic pattern; it says nothing about structure of other forms.
