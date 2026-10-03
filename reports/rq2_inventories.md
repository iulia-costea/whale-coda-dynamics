# RQ2 token inventories (DEC-22)

Each inventory relabels all 2,075 rows of `data/processed/rq1_targets.jsonl` (1,547 comparison targets plus context-only codas) from each coda's own raw intervals and duration in `data/interim/codas_normalized.jsonl`. Only `Ct_raw`, `Ct`, `A_history` and `B_history` change (histories rebuilt from the history IDs), plus an `inventory` field; targets, exchanges, S, row order and coda IDs are RQ1's. Inventory A is `guy_coda_type`, RQ1's own labels. Labels are counted before any fit.

## Rules

- Rhythm (`[R2-1]`): n = nClicks; n < 3 or n > 10 gives r17; otherwise the click positions (cumulative sums of ICI1…ICI(n−1), 0 prepended, divided by their total) take the label of the template in `mean_codas.p` with n clicks that is nearest by mean squared difference (ties: lowest index). Ties among the 2,075 rows: 0. Format `r<k>`.
- Tempo (`[R2-2]`): class j = number of cuts strictly below Duration. KDE cuts: 0.45, 0.61, 0.93, 1.08 s. `tempos-dict.p` cuts (midpoints between adjacent classes): 0.392234, 0.593804, 0.90685, 1.09069 s. Format `t<j>`, combined with `|` (`r2|t4`, `1+1+3|t4`).

## Checks

| Check | Result |
|---|---|
| KDE minima (tempo cuts), s | 0.45, 0.61, 0.93, 1.08 |
| KDE maxima (modes), s | 0.33, 0.51, 0.8, 1.02, 1.26 |
| Rhythm rule vs `rhythms.p`, all `[D2]` rows | 3,840 / 3,840 (ties: 0) |
| Rule on `[D1]`'s intervals vs `rhythms.p`, matched codas | 2,743 / 2,769 |
| KDE vs `tempos-dict.p` tempo class, comparison targets | 1,511 / 1,547 |

- KDE: unnormalised Gaussian kernel (bandwidth 0.035 s) over the 7,737 EC1 durations d with 0.1 < d < 1.75 s in `data/raw/CarribeanCodas.csv`, evaluated at 0.10, 0.11, …, 1.75 s (166 points); strict interior local minima and maxima. The stage raises unless they equal 0.45, 0.61, 0.93, 1.08 and 0.33, 0.51, 0.8, 1.02, 1.26 s, and unless the rhythm rule reproduces every `rhythms.p` label.
- Matching `[D1]` to `[D2]`: one-to-one on (REC, TsTo, Duration) with TsTo and Duration rounded to 3 decimals, because the two releases print the same values differently (`0.9277916000000002` vs `0.9277916`). 2,769 of the 3,933 `[D1]` codas match (0 without a parseable key). Of the 26 that disagree, 26 differ between the two releases in click count or in an interval used by the rule (by more than 1e-06 s).

## Codas with irregular intervals

Retained codas with 3–10 clicks whose first n−1 intervals include a zero, or with non-zero intervals beyond them. The rule is applied to the first n−1 intervals as written; these are parsed values, not missing ones.

| Coda | Comparison target | nClicks | guy_coda_type | ICI1…ICI(n−1) | Non-zero beyond n−1 |
|---|---|---:|---|---|---|
| assayag:row:1635 | yes | 7 | 6i | 0.3751564, 0.3099486, 0.1743554, 0.1221399, 0.2079381, 0.0 | — |
| assayag:row:3366 | yes | 6 | 6i | 0.2628426, 0.0, 0.0, 6.49999856e-06, 0.00955 | 0.2630974, 0.3044469 |
| assayag:row:3606 | yes | 5 | 6R | 0.425535, 0.4157781, 0.1544572, 0.158506 | 6.95 |

## Inventories

Counts on the 1,547 comparison targets unless stated. Repeats: targets equal to the whale's own last label. Labels over all 2,075 rows include context-only codas. "Label ≠ A's" is given for B only: the other inventories use different label strings. Flips: targets that repeat under one of A and the inventory and switch under the other.

| ID | Inventory | File | Labels | Top label | Top share | Repeats | Labels, all rows | Occurring once | Label ≠ A's | Flips vs A |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| A | guy_coda_type (RQ1) | `data/processed/rq1_targets.jsonl` | 21 | `1+1+3` | 57.2% | 1,277 | 21 | 0 | — | — |
| B | decision_tree_coda_type | `data/processed/rq2_b_dtree_targets.jsonl` | 25 | `1+1+3` | 56.9% | 1,280 | 25 | 1 | 75 | 17 |
| C | Sharma rhythm | `data/processed/rq2_c_rhythm_targets.jsonl` | 16 | `r2` | 61.9% | 1,215 | 18 | 2 | — | 74 |
| D | Sharma rhythm × tempo (KDE cuts) | `data/processed/rq2_d_rhythm_tempo_targets.jsonl` | 42 | `r2|t4` | 39.9% | 1,142 | 56 | 13 | — | 143 |
| E | guy_coda_type × tempo (KDE cuts) | `data/processed/rq2_e_type_tempo_targets.jsonl` | 39 | `1+1+3|t4` | 39.9% | 1,204 | 50 | 10 | — | 73 |
| D′ | Sharma rhythm × tempo (tempos-dict.p cuts) | `data/processed/rq2_d_rhythm_tempo_tdict_targets.jsonl` | 47 | `r2|t4` | 39.5% | 1,147 | 58 | 13 | — | 134 |
| E′ | guy_coda_type × tempo (tempos-dict.p cuts) | `data/processed/rq2_e_type_tempo_tdict_targets.jsonl` | 41 | `1+1+3|t4` | 39.4% | 1,210 | 49 | 8 | — | 67 |

UNK counts per fold come from each inventory's fold harness (`reports/rq2_<name>_folds.md`).

## Reproducibility

Input SHA-256:

- `data/processed/rq1_targets.jsonl`: `9c7245c36fcc392dec0e67db08b7d943cd89107eb1b34e3b2b8cc55e1934be02`
- `config/rq1.json`: `44d17a5d082b68fb8cda54b79c4ee563117d59807dacf38d04cd0fb256e4581a`
- `data/interim/codas_normalized.jsonl`: `cf7d636ab0604c7e239d8678edcc88208fd15738273da9fa36d5f9dfaf50f0ff`
- `data/interim/sharma_inventory.json`: `071ba722b03676f432565b79232550e951f04dc5e35d4044e44bcebccc1f6093`
- `data/raw/CarribeanCodas.csv`: `c8ad7b47e6bc102db5a40d7dd2bbbebe8026e4271d135a8e952d68f1baddebd5`
- `data/raw/sperm-whale-dialogues.csv`: `7ca90b5387aea6ec1517f5a66400399093f5df69b09badcba3ba084ace845ac3`
- `data/raw/rhythms.p`: `cd204bef55e7c296ac7f035babfc71b0e34ee8f81ad286c3e430186bc9f2238b`

Output SHA-256:

- `data/processed/rq2_b_dtree_targets.jsonl`: `be16e65abda3ee4a13bf7fc3b81e4265d7f8ad5c6639cd41920fcaf872cbe0ec`
- `data/processed/rq2_c_rhythm_targets.jsonl`: `a315defd8ae4c63f4f47963138e26436b34110b44bcfd1e52a2274b71dceb090`
- `data/processed/rq2_d_rhythm_tempo_targets.jsonl`: `4b49578dfaab640d9b278b18ebe58f03331616e52faf70ddb6a3083febdc05e5`
- `data/processed/rq2_e_type_tempo_targets.jsonl`: `dbfb2fda49169aaa60e01a8e059f0d44ddce921118ee6cd0d1af268c2643c7f2`
- `data/processed/rq2_d_rhythm_tempo_tdict_targets.jsonl`: `154eac17a3ea34bbe27f66d3c50b7019b1c62df4c5ee5262b6a7f190fdafa9d6`
- `data/processed/rq2_e_type_tempo_tdict_targets.jsonl`: `bd54f96b8e6a0e78f8da68e7f6f10bf7cfb13da21b1b6e756616b81feb7bbeb4`
