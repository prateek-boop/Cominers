# Training remediation — 10 September 2026

The diagnosed graph-gradient issue has an opt-in fix, and its numerical behavior
and inference compatibility are tested. This does not establish production
accuracy. Controlled feature-model comparisons are recorded separately in
`controlled-baselines.json`; their test split is deliberately unopened.

## Numerical diagnosis and fix

The first 4,096 CIC2017 training rows were replayed with the same seed,
initialization, batch size, and optimizer. The original time encoding reached a
pre-clipping gradient norm of 11,770,514, dominated by the memory time encoder's
linear weights. Newly seen nodes have last-update time zero, making initial
time deltas comparable to Unix timestamps. Applying signed log1p to time deltas
reduced the maximum norm to 7.8035 and clipped steps from 90/128 to 10/128.

The new transform is explicitly recorded in checkpoints and read by training,
ordinary sequence inference, streaming inference, and stateful inference.
Existing checkpoints default to their original identity transform. No existing
checkpoint was converted or promoted. This probe covers a short training prefix,
not all traffic or attack types; lower gradient norms do not prove higher F1.

Evidence: `graph-gradient-probe.json`; implementation: `models/time_features.py`;
reproduction: `scripts/diagnose_graph_gradients.py`.

## Data findings

A seed-42 uniform priority sample, capped at 512 rows per family/capture/split,
contained 14,240 training/validation rows. All eligible forward/backward packet
mean and flow byte/packet rate checks agreed within 1% relative / 0.01 absolute
tolerance. Backward-mean checks excluded zero backward-packet counts. These checks
support the selected columns' arithmetic consistency, not the correctness of
every feature or label in the complete datasets.

Of 11,618 distinct sampled float32 log-feature vectors, 182 had both benign and
attack labels somewhere in the sample. This is evidence of ambiguity in the
16-feature representation; it does not prove those source labels are wrong.
Stratified sampling prevents treating 182/11,618 as a population error rate or
an accuracy ceiling. No source rows were relabeled or deleted on this basis.
Conflicts also occur within individual capture days, including 10 sampled
vectors on IDS2018 February 23, 17 on March 1, and four on CIC2017 Thursday.
Thus cross-dataset differences alone do not explain all observed ambiguity.

IDS2018 training has only 249 web-brute-force, 79 XSS, and 34 SQL-injection rows,
compared with 686,012 HOIC examples. More rows from dominant families do not
resolve this shortage. Family weighting is tested as a training intervention;
it cannot supply missing independent examples.

Evidence and complete counts: `feature-consistency.json` and each controlled
run's `run.json`.

## Verification and limits

The full suite passed 88 tests. Focused tabular tests additionally passed after
adding assertions for validation-only holdout isolation, CIC2017-only training
selection, unchanged validation coverage, and train-only normalization.

The comparisons use one seed, one architecture and a fixed maximum epoch budget;
larger datasets receive more optimizer updates per epoch. They diagnose this
training policy and do not establish statistical superiority across seeds.
The deployment checkpoint SHA256 remains
`710e6d00265fe2c2de2f7a6a38f1045c483b80a718ba9aec588e3bf526507a79`.

## Completed controlled comparisons

All three runs completed with early stopping after seven epochs; epoch four was selected in each run. No test data was evaluated.

The following are validation diagnostics at the predeclared threshold 0.5, not approved operating points. AP summarizes ranking across thresholds; it is not accuracy.

| Training data / weighting | AP | Precision | Recall | F1 | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| cic2017 | 0.054645 | 0.099142 | 0.004612 | 0.008814 | 0.002375 |
| combined | 0.101922 | 0.090288 | 0.268249 | 0.135103 | 0.153187 |
| family | 0.103567 | 0.110141 | 0.356299 | 0.168266 | 0.163154 |

None found a threshold satisfying precision >=95%, recall >=95%, and false-positive rate <=0.1%, with capture-level checks. The saved fallback threshold of 1.0 is a failed-feasibility marker; its near-zero detection metrics are not the fixed-0.5 diagnostics above.

Adding IDS2018 improved aggregate AP from 0.054645 to 0.101922 in this matched-architecture, fixed-epoch policy. Family weighting raised AP to 0.103567 and fixed-0.5 F1 to 0.168266, while increasing the false-positive rate to 0.163154. Neither intervention closes the operational gap.

| Validation capture | CIC2017-only AP | Combined AP | Family-weighted AP |
| --- | ---: | ---: | ---: |
| ids2018-Friday-23-02-2018 | 0.000475 | 0.001103 | 0.001202 |
| ids2018-Thursday-01-03-2018 | 0.273324 | 0.345946 | 0.347810 |
| cic2017-thursday | 0.017221 | 0.004857 | 0.006374 |

CIC2017 Thursday AP worsened with the combined feature model even though aggregate AP improved. Capture-level reporting is therefore necessary; the overall score cannot establish reliable generalization.

Decision: do not promote any of these checkpoints or treat weighting as the solution. Before a larger production-oriented run, address the observed feature ambiguity and rare-family shortage with independently audited examples and candidate features that can distinguish the conflicting flows. Evaluate the signed-log graph candidate separately; this turn established its numerical behavior, not its detection accuracy.
