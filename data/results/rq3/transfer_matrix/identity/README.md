# Pacific identity-status breakdown

`scores.csv` attaches Hersh et al. Table S2 `identity`/`non_identity` labels to existing Pacific target-coda paired scores. Models were trained on all eligible codas; this is a post-scoring analysis. `bootstrap_draws.csv` and `summary.csv` provide stratum-specific uncertainty.

| Status | Pacific↔Pacific directed cells | Mean transfer loss (bits/symbol) |
|---|---:|---:|
| identity | 42 | 1.789521 |
| non_identity | 42 | 0.979747 |

Non-identity codas have lower average transfer loss than identity codas in the Pacific↔Pacific subset. This descriptive contrast is distinct from the overlap test below.
