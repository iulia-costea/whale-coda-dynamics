# RQ2 inventory B, decision_tree_coda_type: leave-one-day-out folds

14 folds; 1547 comparison targets, each assigned to test exactly once. No predictions or models yet.
Each fold has one test day, one validation day, and all remaining days for training. Validation is the next day in a seeded hash-ordered ring; each day is validation once. This rule does not inspect labels or outcomes.
Original labels are recovered from Ct_raw and history IDs. Vocabulary counts all retained training codas once, including context-only codas. Priors use training comparison targets only. No numerical transformations are fitted at this stage.
All eight conditions share each fold's targets, vocabulary and partitions. Preparation's original fixed-split tables remain unchanged.

| Test day | Validation day | Train targets | Validation targets | Test targets | Vocabulary incl. UNK | Train UNK | Validation UNK | Test UNK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2014-04-16 | 2015-04-01 | 1361 | 126 | 60 | 24 | 1 | 0 | 1 |
| 2015-04-01 | 2015-04-10 | 1337 | 84 | 126 | 25 | 1 | 0 | 0 |
| 2015-04-10 | 2016-05-13 | 1433 | 30 | 84 | 25 | 1 | 0 | 0 |
| 2016-05-13 | 2014-04-24 | 1489 | 28 | 30 | 25 | 1 | 0 | 0 |
| 2014-04-24 | 2015-04-07 | 1488 | 31 | 28 | 25 | 1 | 0 | 0 |
| 2015-04-07 | 2016-05-12 | 1402 | 114 | 31 | 25 | 1 | 0 | 0 |
| 2016-05-12 | 2014-04-29 | 1385 | 48 | 114 | 24 | 1 | 1 | 0 |
| 2014-04-29 | 2015-03-19 | 1433 | 66 | 48 | 24 | 1 | 0 | 1 |
| 2015-03-19 | 2016-04-24 | 1465 | 16 | 66 | 25 | 1 | 0 | 0 |
| 2016-04-24 | 2016-04-12 | 1479 | 52 | 16 | 25 | 1 | 0 | 0 |
| 2016-04-12 | 2015-03-02 | 1141 | 354 | 52 | 21 | 4 | 28 | 30 |
| 2015-03-02 | 2015-03-31 | 911 | 282 | 354 | 23 | 3 | 0 | 24 |
| 2015-03-31 | 2015-03-26 | 1009 | 256 | 282 | 25 | 0 | 1 | 0 |
| 2015-03-26 | 2014-04-16 | 1231 | 60 | 256 | 24 | 0 | 1 | 1 |

Validated: original histories and timing recomputed; day/tag/exchange/context separation; mapped labels independently recomputed; nonempty partitions; exact once-only test coverage.
Manifest: `data/processed/rq2_b_dtree_folds/manifest.json`. Contains target IDs, configurations, raw training counts, priors, and target/history UNK counts for every fold.
Input SHA-256: `be16e65abda3ee4a13bf7fc3b81e4265d7f8ad5c6639cd41920fcaf872cbe0ec`.
