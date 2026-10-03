# RQ2 inventory D, Sharma rhythm × tempo (KDE cuts): leave-one-day-out folds

14 folds; 1547 comparison targets, each assigned to test exactly once. No predictions or models yet.
Each fold has one test day, one validation day, and all remaining days for training. Validation is the next day in a seeded hash-ordered ring; each day is validation once. This rule does not inspect labels or outcomes.
Original labels are recovered from Ct_raw and history IDs. Vocabulary counts all retained training codas once, including context-only codas. Priors use training comparison targets only. No numerical transformations are fitted at this stage.
All eight conditions share each fold's targets, vocabulary and partitions. Preparation's original fixed-split tables remain unchanged.

| Test day | Validation day | Train targets | Validation targets | Test targets | Vocabulary incl. UNK | Train UNK | Validation UNK | Test UNK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2014-04-16 | 2015-04-01 | 1361 | 126 | 60 | 40 | 3 | 1 | 1 |
| 2015-04-01 | 2015-04-10 | 1337 | 84 | 126 | 42 | 2 | 0 | 1 |
| 2015-04-10 | 2016-05-13 | 1433 | 30 | 84 | 43 | 2 | 1 | 0 |
| 2016-05-13 | 2014-04-24 | 1489 | 28 | 30 | 43 | 2 | 0 | 1 |
| 2014-04-24 | 2015-04-07 | 1488 | 31 | 28 | 43 | 2 | 0 | 0 |
| 2015-04-07 | 2016-05-12 | 1402 | 114 | 31 | 43 | 2 | 0 | 0 |
| 2016-05-12 | 2014-04-29 | 1385 | 48 | 114 | 43 | 2 | 2 | 0 |
| 2014-04-29 | 2015-03-19 | 1433 | 66 | 48 | 43 | 2 | 0 | 2 |
| 2015-03-19 | 2016-04-24 | 1465 | 16 | 66 | 44 | 2 | 0 | 0 |
| 2016-04-24 | 2016-04-12 | 1479 | 52 | 16 | 42 | 2 | 4 | 0 |
| 2016-04-12 | 2015-03-02 | 1141 | 354 | 52 | 33 | 6 | 34 | 31 |
| 2015-03-02 | 2015-03-31 | 911 | 282 | 354 | 36 | 6 | 2 | 31 |
| 2015-03-31 | 2015-03-26 | 1009 | 256 | 282 | 40 | 2 | 5 | 2 |
| 2015-03-26 | 2014-04-16 | 1231 | 60 | 256 | 39 | 4 | 1 | 4 |

Validated: original histories and timing recomputed; day/tag/exchange/context separation; mapped labels independently recomputed; nonempty partitions; exact once-only test coverage.
Manifest: `data/processed/rq2_d_rhythm_tempo_folds/manifest.json`. Contains target IDs, configurations, raw training counts, priors, and target/history UNK counts for every fold.
Input SHA-256: `4b49578dfaab640d9b278b18ebe58f03331616e52faf70ddb6a3083febdc05e5`.
