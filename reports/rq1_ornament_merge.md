# RQ1 ornament merge: inventory O vs A (PLAN [P7], DEC-31)

DEC-28's switch-structure runner (`switch_structure.py`) rerun on inventory O (`ornament_merge.py --stage build`, `reports/rq2_o_ornament_merged.md`) and compared with the committed inventory-A run (`data/results/rq1_switch_structure/PD/scored.jsonl` etc.), condition `A_B_S` only.

## Primary (DEC-31): LL(PD) - LL(R) in A_B_S on O

Total: 0.03601 bits/coda (unweighted day mean 0.03840, 95% CI [0.00291, 0.07810], 12/14 days positive, two-sided sign-flip p = 0.0726).

Repeat part: 0.01214 bits/coda (unweighted day mean 0.01057, 95% CI [-0.00431, 0.02700], 11/14 days positive).

Switch part: 0.02387 bits/coda (unweighted day mean 0.02783, 95% CI [0.00183, 0.05604], 9/14 days positive).

## Change from inventory A

[LL_O(PD) - LL_O(R)] - [LL_A(PD) - LL_A(R)], paired by coda: -0.01569 bits/coda (unweighted day mean -0.02331, 95% CI [-0.04104, 0.00203], 5/14 days positive, two-sided sign-flip p = 0.1287).

## PD - R gain split

| Inventory | Part | Count | Share of gain | Mean per target |
|---|---|---:|---:|---:|
| O | return_after_switch | 34 | 65.8% | 1.07764 |
| O | other_after_switch | 151 | -15.8% | -0.05818 |
| O | other | 1362 | 50.0% | 0.02045 |
| A | return_after_switch | 61 | 76.9% | 1.00846 |
| A | other_after_switch | 174 | -23.0% | -0.10552 |
| A | other | 1312 | 46.0% | 0.02807 |

## LL(PD) - LL(P)

O: 0.01290 bits/coda (unweighted day mean 0.00688, 95% CI [-0.00100, 0.02591], 8/14 days positive, two-sided sign-flip p = 0.1433).

A: 0.01310 bits/coda (unweighted day mean 0.01214, 95% CI [0.00248, 0.02670], 10/14 days positive, two-sided sign-flip p = 0.0525).

## O's v4 lattice beside RQ1's

From each run's own logistic v4 evaluation output (not the switch-structure runner).

| Contrast | O | RQ1 |
|---|---|---|
| own (A_B_S - B_S) | 1.49478 bits/coda (unweighted day mean 1.46852, 95% CI [1.05004, 1.94382], 14/14 days positive) | 1.34369 bits/coda (unweighted day mean 1.25568, 95% CI [0.94454, 1.73171], 14/14 days positive) |
| partner (A_B_S - A_S) | -0.02829 bits/coda (unweighted day mean -0.01999, 95% CI [-0.07530, 0.02644], 7/14 days positive) | -0.02566 bits/coda (unweighted day mean -0.00144, 95% CI [-0.09422, 0.06351], 6/14 days positive) |
| timing (A_B_S - A_B) | -0.01619 bits/coda (unweighted day mean -0.00286, 95% CI [-0.03107, 0.00337], 10/14 days positive) | -0.02024 bits/coda (unweighted day mean -0.00885, 95% CI [-0.04635, 0.00136], 7/14 days positive) |

## Gate (O's R reproducing logistic v4)

Passed: True.

## Stop rules

| Run | Variant | DEC-16 fires | DEC-23 fires |
|---|---|---|---|
| O | R | 0 | [] |
| O | L | 0 | [] |
| O | P | 0 | [] |
| O | D | 0 | [] |
| O | PD | 0 | [] |
| A | R | 0 | [] |
| A | L | 0 | [] |
| A | P | 0 | [] |
| A | D | 0 | [] |
| A | PD | 0 | [] |

## Reading

- Primary null: the gain isn't detectable once flagged ornaments are merged; DEC-28's secondary is not evidence of switch-backs beyond ornaments (the interval is above zero, but the two-sided sign-flip p = 0.0726 is not below 0.05).
- Change from A spans zero: not detectably different from DEC-28's inventory-A gain.

## Reproducibility

Input SHA-256:

- `data/results/rq1_switch_structure_ornament_merged/summary.json`: `39053ef8e0e8af35513850b03c860d56c80700b209bd00c98eee84694cc51539`
- `data/results/rq1_switch_structure/summary.json`: `0fc3b9f0540d06e3c1f9f02a672344a48192ec2a14d3caa7a16888cc1c181f1b`
- `data/results/rq2_o_ornament_merged_logistic_v4/evaluation/summary.json`: `9331061f25f8b54556f16cea36e8e815e1be66c05c2c3266d40f28c0fe9cd92f`
- `data/results/rq1_logistic_v4/evaluation/summary.json`: `b782c2705b001a346d350c6043b725b047bf2bb6c7e0a801546d33fa0b998980`
- `data/results/rq1_switch_structure_ornament_merged/R/scored.jsonl`: `6dd1ccf25251c9509ce06910f1493de7307985f6a2fc7cb2648152b9b7f9644a`
- `data/results/rq1_switch_structure_ornament_merged/P/scored.jsonl`: `605d35a2726090f0f88f4afa634ccae6f457cdcc4c5c79e3732e417014366007`
- `data/results/rq1_switch_structure_ornament_merged/PD/scored.jsonl`: `f68d15c6105817d0740d209b34282561ac626021d20b6ed50095718738e2ef17`
- `data/results/rq1_switch_structure/R/scored.jsonl`: `765e1c729c5f9661bbadd15ce472ba326c0624bc4ac9ecdb9208e042ab6eb3bf`
- `data/results/rq1_switch_structure/P/scored.jsonl`: `a8450d47a73ef538c455cab024404b0fa43674e7672dba5bb851af120994dad9`
- `data/results/rq1_switch_structure/PD/scored.jsonl`: `f4df23f3ce2edc210840e246ccefba3d011c7d25e529e9e42e923ccc73a2c4e5`
- `data/processed/rq2_o_ornament_merged_targets.jsonl`: `da5ace8a3c300831157c0967ca4cf19863760c7734c672fdf288a70ddcf4fa06`
- `data/processed/rq1_targets.jsonl`: `9c7245c36fcc392dec0e67db08b7d943cd89107eb1b34e3b2b8cc55e1934be02`