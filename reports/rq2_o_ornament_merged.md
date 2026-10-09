# RQ2 inventory O: ornament merge (PLAN [P7], DEC-31)

`guy_coda_type` (inventory A) with every coda Sharma et al.'s released flags (`data/raw/ornaments.p`, github.com/pratyushasharma/sw-combinatoriality commit `7228c8eed2cc27ddd23b74c51aeccec9d762389e`) mark as ornamented relabelled to its neighbour's inventory-A type. Targets, exchanges, S, row order, folds and the UNK policy are RQ1's; only `Ct_raw`, `Ct`, `A_history` and `B_history` change (histories rebuilt from the history IDs), plus an `inventory` field.

## Matching and flagging

`[D1]` matched one-to-one to `[D2]` on (REC, TsTo, Duration) rounded to 3 decimals, as `inventories.check_d1_agreement` does.

| Check | Result |
|---|---|
| RQ1 rows (context-only codas included) | 2,075 |
| Matched one-to-one to [D2] | 1,706 (0 without a parseable key) |
| Matched rows flagged ornamented | 66 |
| Flagged with a qualifying neighbour (preceding / following) | 55 (27 / 28) |
| Flagged without a qualifying neighbour (kept) | 11 |
| Chosen neighbours that are themselves flagged | 0 |
| Rows whose label changes | 51 |
| Comparison targets whose label changes | 31 |

Neighbour rule: among the whale's immediately preceding and following codas in the same exchange (ordered by onset), those within 10 s with exactly one fewer click, the nearer; a tie goes to the preceding one. Without a qualifying neighbour the coda keeps its label.

## Flagged codas

| Coda | Old label | New label | Neighbour | Side |
|---|---|---|---|---|
| assayag:row:102 | `6i` | `1+1+3` | assayag:row:100 | preceding |
| assayag:row:119 | `10i` | `10i` | assayag:row:117 | preceding |
| assayag:row:1271 | `8R` | `7D2` | assayag:row:1273 | following |
| assayag:row:133 | `6i` | `5R1` | assayag:row:135 | following |
| assayag:row:138 | `6i` | `5R1` | assayag:row:141 | following |
| assayag:row:1432 | `1+1+3` | `4R2` | assayag:row:1433 | following |
| assayag:row:148 | `6i` | `5R1` | assayag:row:147 | preceding |
| assayag:row:1506 | `6i` | `5R1` | assayag:row:1507 | following |
| assayag:row:1511 | `6R` | `1+1+3` | assayag:row:1512 | following |
| assayag:row:1596 | `6i` | `1+1+3` | assayag:row:1597 | following |
| assayag:row:1635 | `6i` | `6i` | — | kept: no neighbour |
| assayag:row:1656 | `6R` | `1+1+3` | assayag:row:1654 | preceding |
| assayag:row:1673 | `10i` | `10i` | — | kept: no neighbour |
| assayag:row:1728 | `7D2` | `7D2` | — | kept: no neighbour |
| assayag:row:1734 | `8R` | `7i` | assayag:row:1735 | following |
| assayag:row:1849 | `8R` | `7D2` | assayag:row:1846 | preceding |
| assayag:row:1872 | `6i` | `5R2` | assayag:row:1874 | following |
| assayag:row:197 | `6R` | `1+1+3` | assayag:row:198 | following |
| assayag:row:199 | `6R` | `1+1+3` | assayag:row:198 | preceding |
| assayag:row:20 | `5R3` | `4R2` | assayag:row:18 | preceding |
| assayag:row:2082 | `6i` | `1+1+3` | assayag:row:2083 | following |
| assayag:row:209 | `6i` | `1+1+3` | assayag:row:207 | preceding |
| assayag:row:212 | `8R` | `7D2` | assayag:row:214 | following |
| assayag:row:2138 | `8R` | `7D2` | assayag:row:2136 | preceding |
| assayag:row:2316 | `7i` | `7i` | — | kept: no neighbour |
| assayag:row:2376 | `6i` | `1+1+3` | assayag:row:2378 | following |
| assayag:row:2411 | `1+1+3` | `4R2` | assayag:row:2409 | preceding |
| assayag:row:2423 | `8R` | `7D2` | assayag:row:2421 | preceding |
| assayag:row:249 | `6i` | `5R1` | assayag:row:252 | following |
| assayag:row:2546 | `6i` | `6i` | — | kept: no neighbour |
| assayag:row:2606 | `10i` | `10i` | assayag:row:2605 | preceding |
| assayag:row:2698 | `7i` | `7i` | — | kept: no neighbour |
| assayag:row:2706 | `6i` | `5R1` | assayag:row:2708 | following |
| assayag:row:2746 | `6i` | `5R1` | assayag:row:2747 | following |
| assayag:row:2839 | `6i` | `5R1` | assayag:row:2837 | preceding |
| assayag:row:2863 | `6i` | `5R1` | assayag:row:2864 | following |
| assayag:row:2866 | `8R` | `7i` | assayag:row:2867 | following |
| assayag:row:287 | `6i` | `1+1+3` | assayag:row:286 | preceding |
| assayag:row:2922 | `6i` | `5R1` | assayag:row:2921 | preceding |
| assayag:row:3303 | `1+1+3` | `4R2` | assayag:row:3302 | preceding |
| assayag:row:3325 | `9i` | `8R` | assayag:row:3327 | following |
| assayag:row:3329 | `5R3` | `4R2` | assayag:row:3331 | following |
| assayag:row:3350 | `10i` | `10i` | assayag:row:3349 | preceding |
| assayag:row:3362 | `6R` | `1+1+3` | assayag:row:3361 | preceding |
| assayag:row:3366 | `6i` | `6i` | — | kept: no neighbour |
| assayag:row:3387 | `8R` | `8R` | — | kept: no neighbour |
| assayag:row:3388 | `6i` | `5R1` | assayag:row:3389 | following |
| assayag:row:3474 | `5R1` | `4R1` | assayag:row:3476 | following |
| assayag:row:3487 | `6i` | `6i` | — | kept: no neighbour |
| assayag:row:3488 | `8R` | `7i` | assayag:row:3489 | following |
| assayag:row:37 | `10i` | `10i` | assayag:row:36 | preceding |
| assayag:row:3789 | `10R` | `10R` | — | kept: no neighbour |
| assayag:row:3866 | `6i` | `5R1` | assayag:row:3864 | preceding |
| assayag:row:3880 | `6i` | `5R1` | assayag:row:3878 | preceding |
| assayag:row:3892 | `10R` | `9R` | assayag:row:3895 | following |
| assayag:row:3917 | `9i` | `8R` | assayag:row:3915 | preceding |
| assayag:row:44 | `10i` | `10i` | — | kept: no neighbour |
| assayag:row:53 | `6R` | `1+1+3` | assayag:row:52 | preceding |
| assayag:row:54 | `9i` | `8R` | assayag:row:55 | following |
| assayag:row:59 | `6i` | `5R1` | assayag:row:57 | preceding |
| assayag:row:71 | `6i` | `5R1` | assayag:row:70 | preceding |
| assayag:row:79 | `6i` | `5R1` | assayag:row:77 | preceding |
| assayag:row:81 | `6i` | `5R1` | assayag:row:82 | following |
| assayag:row:86 | `6i` | `5R1` | assayag:row:84 | preceding |
| assayag:row:87 | `6i` | `1+1+3` | assayag:row:88 | following |
| assayag:row:9 | `1+1+3` | `4R2` | assayag:row:11 | following |

## Inventory statistics (comparison targets unless stated)

| | A | O |
|---|---:|---:|
| Distinct labels | — | 21 |
| Repeats of own last | 1,277 | 1,337 |
| Own lag 1 ≠ lag 2 | 235 | 185 |
| ...of those, target = own lag 2 (return right after a switch) | 61 | 34 |

Repeat/switch flips vs A: 60. Most frequent changes: `6i`→`5R1` 18, `6i`→`1+1+3` 7, `6R`→`1+1+3` 6, `8R`→`7D2` 5, `1+1+3`→`4R2` 4.

Comparison targets: 1,547.

## Reproducibility

Input SHA-256:

- `data/processed/rq1_targets.jsonl`: `9c7245c36fcc392dec0e67db08b7d943cd89107eb1b34e3b2b8cc55e1934be02`
- `data/interim/codas_normalized.jsonl`: `cf7d636ab0604c7e239d8678edcc88208fd15738273da9fa36d5f9dfaf50f0ff`
- `data/raw/sperm-whale-dialogues.csv`: `7ca90b5387aea6ec1517f5a66400399093f5df69b09badcba3ba084ace845ac3`
- `data/raw/ornaments.p`: `8e37ba3e6ee9938286cc255be0a5fd07067fd8aa4b0393c8978e9968c1655f1d`

Output SHA-256:

- `data/processed/rq2_o_ornament_merged_targets.jsonl`: `da5ace8a3c300831157c0967ca4cf19863760c7734c672fdf288a70ddcf4fa06`
