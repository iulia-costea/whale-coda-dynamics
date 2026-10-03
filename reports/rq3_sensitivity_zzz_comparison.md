# RQ3 EC2 ZZZ sensitivity

EC2 codas with `source_unit == ZZZ` were excluded before fold construction and model selection. The sensitivity run therefore has five EC2 one-day folds.

| Source | Target | Primary loss (bits/symbol) | ZZZ-excluded loss (bits/symbol) | Change |
|---|---|---:|---:|---:|
| EC1 | EC2 | 1.702436 | 1.761078 | 0.058642 |
| EC2 | EC1 | 2.320742 | 2.360515 | 0.039773 |
| EC2 | EC2 | 0.000000 | 0.000000 | 0.000000 |
| EC2 | FP | 1.499752 | 1.511396 | 0.011644 |
| EC2 | PALI | 2.123638 | 2.109845 | -0.013792 |
| EC2 | PO | 1.662875 | 1.642440 | -0.020435 |
| EC2 | REG | 1.071307 | 1.057608 | -0.013700 |
| EC2 | RI | 1.730431 | 1.724925 | -0.005506 |
| EC2 | SH | 1.996870 | 1.999864 | 0.002994 |
| EC2 | SI | 2.454268 | 2.441254 | -0.013014 |
| FP | EC2 | 1.069505 | 1.117637 | 0.048132 |
| PALI | EC2 | 1.966420 | 2.004445 | 0.038025 |
| PO | EC2 | 1.122532 | 1.150980 | 0.028448 |
| REG | EC2 | 0.406145 | 0.370903 | -0.035242 |
| RI | EC2 | 1.494400 | 1.509239 | 0.014839 |
| SH | EC2 | 1.122868 | 1.167207 | 0.044339 |
| SI | EC2 | 0.792468 | 0.829735 | 0.037267 |

Positive change means the ZZZ-excluded transfer loss is larger.
