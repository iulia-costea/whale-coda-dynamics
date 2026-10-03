# Coverage diagnostic

`summary.csv` splits each cross-clan source-fold/target-coda prediction by whether the target's complete fixed-token timing sequence occurred at least five times in that source fold's actual training codas.

Across 1,289,693 evaluations, 7.9% are covered and 92.1% are uncovered.
Covered patterns have lower mean loss in 68 of 72 directed pairs. This indicates that source inventories lacking a target's exact full timing pattern explain a substantial part of poor transfer.
