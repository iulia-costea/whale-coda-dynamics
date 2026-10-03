# RQ1 residual LSTM: device and throughput (step 6d)

Fold test-2015-03-02, condition A_B_S, every block on, hidden 64, dropout 0.0, normal early stopping (patience 20). Each worker fits seeds [42, 43, 44] once; throughput = fits / wall time, workers started beforehand. CPU workers use one torch thread each. The GPU is eligible only if two fits with the same seed, in two processes under torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG (here ':4096:8'), give identical curves and weights. Ties within 5% go to the CPU.

| Device | Processes | Fits | Wall seconds | Fits/hour | Mean epochs/fit | Eligible |
|---|---:|---:|---:|---:|---:|---|
| cpu | 18 | 54 | 37.7 | 5150.8 | 28.3 | yes |
| cuda | 1 | 3 | 8.6 | 1257.6 | 28.3 | yes |
| cuda | 2 | 6 | 9.8 | 2206.7 | 28.3 | yes |
| cuda | 4 | 12 | 13.7 | 3162.0 | 28.3 | yes |

| Determinism check | Seed | Identical | Note |
|---|---:|---|---|
| cpu | 42 | True |  |
| cuda | 42 | True |  |

Chosen: cpu with 18 processes (5150.8 fits/hour on the largest candidate). Full run: 11970 fits; estimated wall time 2.3 hours (the largest candidate is the slowest per epoch, so this is conservative per epoch; epochs per fit vary).

