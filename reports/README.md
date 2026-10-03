# Reports index

Which report is current and which is kept only as a record. All reports here except this file and `data_dictionary.md` are **generated**: edit the code that writes them, never the markdown.

| Report | What | Status |
|---|---|---|
| `rq1.md` | RQ1 preprocessing: counts, exclusions, $S$ summary, UNK policy, checksums | Current |
| `rq2.md` | Sharma-pipeline preprocessing (a robustness check on the whole chain, **not** RQ2 results) | Current |
| `rq1_folds.md` | The 14 leave-one-day-out folds, vocabulary and UNK counts per fold | Current |
| `rq1_folds_retrospective.md` | The same folds for the old-delay sensitivity run | Current |
| `rq1_logistic.md` | Logistic regression v1: fixed L2 = 0.01, no copy weights | Superseded by v2 (DEC-16); kept as a record |
| `rq1_logistic_v2.md` | Logistic regression v2: copy weights, one shared L2 chosen on the validation day | Superseded by v4 (DEC-16, DEC-20); kept as a record |
| `rq1_retrospective_sensitivity.md` | Old-delay (retrospective) sensitivity run, on the v2 model | Superseded by the v4 retrospective run (DEC-20); kept as a record |
| `rq1_logistic_v3.md` | Logistic regression v3: copy weights, one shared L2, grid extended to 0.0001, DEC-16 log-likelihood-only stop rules | Superseded by v4 (DEC-20): gate 1 held except one stop rule (`add_input:A_B_S-A_B`), traced to the shared history/S penalty; kept as a record |
| `rq1_retrospective_sensitivity_v3.md` | Old-delay sensitivity run, on the v3 model | Superseded by the v4 retrospective run (DEC-20); kept as a record |
| `rq1_logistic_v4.md` | Logistic regression v4: copy weights, separate validation-chosen L2 for history and for $S$ (DEC-20) | Current reference. Gate 1 fully holds: no DEC-16 stop rule fires, all fits converged |
| `rq1_retrospective_sensitivity_v4.md` | Old-delay sensitivity run, on the v4 model | Current |
| `rq1_lstm.md` | LSTM v1: fixed WhaleLM settings, no copy weights | Superseded by DEC-17; kept as a record of why |
| `rq1_lstm_reproduction.md` | Assayag partner-only reproduction, v1 | Superseded: reframed as a context check in DEC-17 |
| `rq1_lstm_v2_pilot.md` | LSTM v2 pilot (DEC-17, DEC-20): conditions A and A_B_S on four test days | Superseded by the residual LSTM (DEC-21); kept as a record |
| `rq1_lstm_v2_suppression.md` | LSTM v2 S-suppression diagnostic (DEC-20) | Superseded by DEC-21; kept as a record |
| `rq1_lstm_v2_timing.md` | LSTM v2 timing and determinism check | Superseded by DEC-21; kept as a record |
| `rq1_lstm_residual_checks.md` | Residual LSTM checks before the full run, training and validation days only: wiring, learnability, planted signal (DEC-21, PLAN `[M4]`) | Current. All gates pass |
| `rq1_lstm_residual_throughput.md` | Residual LSTM device and throughput measurement; chose CPU ×18 | Current |
| `rq1_lstm_residual.md` | Residual LSTM on top of a frozen logistic v4 (DEC-21): the enlarged-family check, both lattices, residual gains, chosen candidates, Assayag context check | Current. No stop rule fires |
| `rq1_lstm_residual_sensitivity.md` | Planted-signal sensitivity run for the residual LSTM's condition A: lag-6 own-coda copies planted on all 14 folds, synthetic labels only (protocol after Bastings et al. 2022) | Current. Gives no bound: the recovery ratio's lower bound is below zero at every f |
| `rq2_inventories.md` | RQ2 token inventories B–E′ (DEC-22): the rule checks against Sharma et al.'s files and the `[D6]` KDE, `[D1]`/`[D2]` and tempo-class agreement, `[R2-0]` counts, codas with irregular intervals, checksums | Current. Every check and `[R2-0]` count reproduces |
| `rq2_b_dtree_folds.md` | The 14 folds for inventory B (`decision_tree_coda_type`): vocabulary and UNK counts per fold | Current |
| `rq2_c_rhythm_folds.md` | The 14 folds for inventory C (Sharma rhythm): vocabulary and UNK counts per fold | Current |
| `rq2_d_rhythm_tempo_folds.md` | The 14 folds for inventory D (Sharma rhythm × tempo, KDE cuts): vocabulary and UNK counts per fold | Current |
| `rq2_e_type_tempo_folds.md` | The 14 folds for inventory E (`guy_coda_type` × tempo, KDE cuts): vocabulary and UNK counts per fold | Current |
| `rq2_d_rhythm_tempo_tdict_folds.md` | The 14 folds for inventory D′ (Sharma rhythm × tempo, `tempos-dict.p` cuts): vocabulary and UNK counts per fold | Current |
| `rq2_e_type_tempo_tdict_folds.md` | The 14 folds for inventory E′ (`guy_coda_type` × tempo, `tempos-dict.p` cuts): vocabulary and UNK counts per fold | Current |
| `rq2_b_dtree_logistic_v4.md` | Logistic v4 on inventory B (`decision_tree_coda_type`); its log-likelihoods are comparable with no other inventory's | Current. Stop rules fire (`add_input:A_S-A`, `add_input:A_B_S-A_B`): no conclusion drawn from it (PLAN `[R2-7]`) |
| `rq2_c_rhythm_logistic_v4.md` | Logistic v4 on inventory C (Sharma rhythm); its log-likelihoods are comparable with no other inventory's | Current. Stop rules fire (`add_input:B_S-B`): no conclusion drawn from it (PLAN `[R2-7]`) |
| `rq2_d_rhythm_tempo_logistic_v4.md` | Logistic v4 on inventory D (Sharma rhythm × tempo, KDE cuts); its log-likelihoods are comparable with no other inventory's | Current. Stop rules fire (`add_input:A_S-A`, `add_input:B_S-B`): no conclusion drawn from it (PLAN `[R2-7]`) |
| `rq2_e_type_tempo_logistic_v4.md` | Logistic v4 on inventory E (`guy_coda_type` × tempo, KDE cuts); its log-likelihoods are comparable with no other inventory's | Current. Stop rules fire (`add_input:A_S-A`): no conclusion drawn from it (PLAN `[R2-7]`) |
| `rq2_d_rhythm_tempo_tdict_logistic_v4.md` | Logistic v4 on inventory D′ (Sharma rhythm × tempo, `tempos-dict.p` cuts); its log-likelihoods are comparable with no other inventory's | Current. Stop rules fire (`add_input:B_S-B`): no conclusion drawn from it (PLAN `[R2-7]`) |
| `rq2_e_type_tempo_tdict_logistic_v4.md` | Logistic v4 on inventory E′ (`guy_coda_type` × tempo, `tempos-dict.p` cuts); its log-likelihoods are comparable with no other inventory's | Current. No stop rule fires |
| `rq2_inventories_v4.md` | RQ2 comparison across the seven inventories (PLAN `[R2-3]`–`[R2-6]`): `[M2]` contrasts, shares of T with paired differences against A, the sign pattern, the same without 2016-04-12, UNK per fold, the exploratory tempo decomposition | Current. Stop rules fire for B, C, D, E and D′; A and E′ have none |
| `data_dictionary.md` | Field meanings, provenance and open questions (hand-written) | Current |
