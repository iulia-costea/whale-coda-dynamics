# Reports index

Which report is current and which is kept only as a record. All reports here except this file and `data_dictionary.md` are **generated**: edit the code that writes them, never the markdown.

| Report | What | Status |
|---|---|---|
| `rq1.md` | RQ1 preprocessing: counts, exclusions, $S$ summary, UNK policy, checksums | Current |
| `rq2.md` | Sharma-pipeline preprocessing (a robustness check on the whole chain, **not** RQ2 results) | Current |
| `rq1_folds.md` | The 14 leave-one-day-out folds, vocabulary and UNK counts per fold | Current |
| `rq1_folds_retrospective.md` | The same folds for the old-delay sensitivity run | Current |
| `rq1_logistic.md` | Logistic regression v1: fixed L2 = 0.01, no copy weights | Superseded by v2 (DEC-16); kept as a record |
| `rq1_logistic_v2.md` | Logistic regression v2: copy weights, L2 chosen on the validation day | Current reference until rerun with the extended L2 grid (DEC-16). Its accuracy-based stop conditions no longer apply (DEC-16) |
| `rq1_retrospective_sensitivity.md` | Old-delay (retrospective) sensitivity run, on the v2 model | Current |
| `rq1_lstm.md` | LSTM v1: fixed WhaleLM settings, no copy weights | Superseded by DEC-17; kept as a record of why |
| `rq1_lstm_reproduction.md` | Assayag partner-only reproduction, v1 | Superseded: reframed as a context check in DEC-17 |
| `data_dictionary.md` | Field meanings, provenance and open questions (hand-written) | Current |
