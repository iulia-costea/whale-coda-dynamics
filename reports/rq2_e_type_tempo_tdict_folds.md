# RQ2 inventory E′, guy_coda_type × tempo (tempos-dict.p cuts): leave-one-day-out folds

14 folds; 1547 comparison targets, each assigned to test exactly once. No predictions or models yet.
Each fold has one test day, one validation day, and all remaining days for training. Validation is the next day in a seeded hash-ordered ring; each day is validation once. This rule does not inspect labels or outcomes.
Original labels are recovered from Ct_raw and history IDs. Vocabulary counts all retained training codas once, including context-only codas. Priors use training comparison targets only. No numerical transformations are fitted at this stage.
All eight conditions share each fold's targets, vocabulary and partitions. Preparation's original fixed-split tables remain unchanged.

| Test day | Validation day | Train targets | Validation targets | Test targets | Vocabulary incl. UNK | Train UNK | Validation UNK | Test UNK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2014-04-16 | 2015-04-01 | 1361 | 126 | 60 | 40 | 3 | 0 | 0 |
| 2015-04-01 | 2015-04-10 | 1337 | 84 | 126 | 41 | 3 | 0 | 0 |
| 2015-04-10 | 2016-05-13 | 1433 | 30 | 84 | 42 | 3 | 0 | 0 |
| 2016-05-13 | 2014-04-24 | 1489 | 28 | 30 | 42 | 3 | 0 | 0 |
| 2014-04-24 | 2015-04-07 | 1488 | 31 | 28 | 42 | 3 | 0 | 0 |
| 2015-04-07 | 2016-05-12 | 1402 | 114 | 31 | 42 | 3 | 0 | 0 |
| 2016-05-12 | 2014-04-29 | 1385 | 48 | 114 | 41 | 3 | 2 | 0 |
| 2014-04-29 | 2015-03-19 | 1433 | 66 | 48 | 39 | 4 | 1 | 3 |
| 2015-03-19 | 2016-04-24 | 1465 | 16 | 66 | 41 | 3 | 0 | 0 |
| 2016-04-24 | 2016-04-12 | 1479 | 52 | 16 | 41 | 3 | 0 | 0 |
| 2016-04-12 | 2015-03-02 | 1141 | 354 | 52 | 32 | 5 | 32 | 31 |
| 2015-03-02 | 2015-03-31 | 911 | 282 | 354 | 35 | 4 | 2 | 29 |
| 2015-03-31 | 2015-03-26 | 1009 | 256 | 282 | 41 | 1 | 2 | 2 |
| 2015-03-26 | 2014-04-16 | 1231 | 60 | 256 | 41 | 2 | 0 | 1 |

Validated: original histories and timing recomputed; day/tag/exchange/context separation; mapped labels independently recomputed; nonempty partitions; exact once-only test coverage.
Manifest: `data/processed/rq2_e_type_tempo_tdict_folds/manifest.json`. Contains target IDs, configurations, raw training counts, priors, and target/history UNK counts for every fold.
Input SHA-256: `bd54f96b8e6a0e78f8da68e7f6f10bf7cfb13da21b1b6e756616b81feb7bbeb4`.
