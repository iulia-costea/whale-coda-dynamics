# Reports index

Which report is current and which is kept only as a record. All reports here except this file and `data_dictionary.md` are **generated**: edit the code that writes them, never the markdown.

| Report | What | Status |
|---|---|---|
| `rq1.md` | RQ1 preprocessing: counts, exclusions, $S$ summary, UNK policy, checksums | Current |
| `rq2.md` | Sharma-pipeline preprocessing (a robustness check on the whole chain, **not** RQ2 results) | Current |
| `rq1_folds.md` | The 14 leave-one-day-out folds, vocabulary and UNK counts per fold | Current |
| `rq1_folds_retrospective.md` | The same folds for the old-delay sensitivity run | Current |
| `rq1_logistic.md` | Logistic regression v1: fixed L2 = 0.01, no copy weights | Superseded by v2 (DEC-16); kept as a record |
| `rq1_logistic_v2.md` | Logistic regression v2: copy weights, L2 chosen on the validation day | Superseded by v4 (DEC-20); retained with its historical accuracy-based stop messages |
| `rq1_retrospective_sensitivity.md` | Old-delay (retrospective) sensitivity run, on the v2 model | Superseded by v4; kept as a record |
| `rq1_logistic_v3.md` | Extended shared-L2 grid | DEC-16 timing stop rule fired; preserved by explicit instruction |
| `rq1_retrospective_sensitivity_v3.md` | v3 old-delay sensitivity | DEC-16 timing stop rule fired; kept as a record |
| `rq1_logistic_v4.md` | Separately selected history/base and S penalties (DEC-20) | Current reference; numerical gate 1 passes |
| `rq1_retrospective_sensitivity_v4.md` | Independent concurrent v4 sensitivity run from commit `19be209` | Historical contributor run; current reconciled artifacts use the report below |
| `rq1_logistic_v4_retrospective.md` | v4 old-delay sensitivity | Current; numerical gate 1 passes; no-S predictions match primary exactly |
| `rq1_lstm_v2_suppression.md` | Predeclared validation-only S suppression diagnostic | Passed: mean difference 0.0019 bits/coda, tolerance 0.01 |
| `rq1_lstm_v2_pilot.md` | Four-fold CPU pilot with separate decay selection | Specified pilot numerical gate passes; both conditions beat logistic only on the small fold. Full study not run: CUDA unavailable |
| `rq1_lstm.md` | LSTM v1: fixed WhaleLM settings, no copy weights | Superseded by DEC-17; kept as a record of why |
| `rq1_lstm_reproduction.md` | Assayag partner-only reproduction, v1 | Superseded: reframed as a context check in DEC-17 |
| `data_dictionary.md` | Field meanings, provenance and open questions (hand-written) | Current |
