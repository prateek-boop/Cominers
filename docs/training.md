# Retraining CyberTGN

Run the commands below from the repository root. For current results, see the
[remediation report](../reports/audit/training-remediation.md).

Use `train.py` to prepare labeled flows, train a candidate, and evaluate unseen
captures. CIC2017 graph training and CIC2017/IDS2018 feature-model experiments have
completed locally; their detection targets remain unmet. Synthetic tests only
verify software execution, not accuracy. Candidate checkpoints never overwrite `checkpoints/tgn_best.pt`.

See [losses and metric definitions](#losses-metrics-and-candidate-selection) for the objectives,
class weighting, diagnostics, selection criteria and empirical target checks.

## What changed for retraining

The v2 model combines temporal graph logits with a learned **current-flow branch**.
The old architecture scored before consuming current features, making isolated
flow predictions uninformative. New checkpoints can use their current 16 features
on the first prediction. Existing v1 checkpoints retain their behavior.

The pipeline includes disk-backed chronological sorting, explicit label mappings,
file hashes, cross-split duplicate-event rejection, training-only normalization,
class-weighted losses, gradient clipping, validation-based early stopping,
and an independent calibration split. Events are never randomly
shuffled or oversampled inside a capture. Eight stage outputs match the existing
named stages; missing stage labels are masked rather than invented.

All three runtimes load v2 checkpoints: `ModelRuntime`, `StreamingModelRuntime`,
and `StatefulCyberTGN`. V2 timestamps use integer Unix seconds without the legacy
float32 epoch-rounding step. Training and deployment must use the same feature
extractor, timestamp convention, processing batch size and history capacity.

## 1. Install

```bash
venv/bin/python -m pip install -r requirements-training.txt
venv/bin/python train.py --help
```

The existing environment already contains the required packages. To train on a
GPU, install a PyTorch build appropriate for that machine and pass `--device cuda`.
CPU is the default. Seed, library versions, arguments and dataset hashes are saved;
bitwise reproducibility across GPU hardware/software versions is not guaranteed.

## 2. Prepare labeled traffic

See [dataset choices](datasets.md). The importer supports canonical CSV files and
common CICFlowMeter column aliases. Other exporters require a reviewed conversion;
UNSW, Zeek, Argus and CICIoT feature tables are not interchangeable with CIC features.

Required columns: `src_ip`, `dst_ip`, `timestamp`, `label`, and all 16 feature
columns in [schema.csv](../examples/training/schema.csv). IPs are used as graph
identities, never as numerical predictive features. Keep real capture identities
(or consistent pseudonyms expressed as IPs); do not fabricate source/destination
IPs for feature-only CSVs. Missing features are fatal, not zero-filled or shifted.
NaNs, infinities, negative flow statistics and unknown labels are rejected with
file and line information. Audit or re-extract bad rows before training.

Feature units match `GET /model`: duration and IAT in microseconds; rates per
second; packet/byte counts as recorded by the selected exporter. Zero-duration
rates must match the serving extractor's zero-rate convention. Similar column
names alone do not establish extractor equivalence (payload vs frame lengths,
timeouts, FIN handling and rate conventions can differ).

Copy [manifest.json](../examples/training/manifest.json), then edit its paths and
labels. Paths are relative to the manifest. Its example filenames are placeholders,
not automatically created train/test splits. Each `group` is a complete capture
session/day/scenario, occurs exactly once, and belongs to one split. Combine CSV
pieces from the same capture into one source file before preparation. A source
can override columns explicitly:

```json
{
  "group": "capture-2017-07-05",
  "split": "train",
  "path": "captures/wednesday.csv",
  "timestamp_format": "%d/%m/%Y %H:%M:%S",
  "timezone": "UTC",
  "timestamp_kind": "start",
  "columns": {"src_ip": "Source IP", "dst_ip": "Destination IP"}
}
```

Use the documented timezone/date format for that specific export; do not assume
the example format/timezone. With no `timestamp_format`, timestamps must be Unix
seconds. `timestamp_kind: start` adds flow duration to obtain the completion time;
`end` means the supplied time already denotes completion.

Map every label explicitly. `{"attack":1,"stage":null}` trains binary detection
without claiming a MITRE stage. Dataset attack families are not automatically
MITRE ground truth: e.g. a label saying “Botnet” does not prove a specific stage.
Keep uncertain/attempted labels aligned with the dataset authors' labeling rules,
and document that decision. Do not feed label, attack category or attempted-category
columns as numerical features.

Use capture-/campaign-disjoint training, validation, calibration and test sets.
Prefer later captures for testing and reserve a second environment/dataset as an
external test. Start around 60/15/10/15 percent **by complete captures**, adjusting
to available independent captures and class support. This is a planning heuristic,
not a guaranteed optimal split. Training, validation and test require both binary
classes. Calibration needs representative benign traffic. Do not copy the same
capture into multiple splits to make class counts look balanced.

```bash
venv/bin/python train.py prepare \
  --manifest examples/training/manifest.json \
  --output data/prepared --shard-size 100000
```

Preparation uses SQLite for sorting and duplicate detection and writes NumPy
shards plus `dataset.json`. It requires free temporary disk space and refuses to
overwrite an output directory. Same-split identical events with matching labels
are deduplicated; cross-split or conflicting copies fail. This checks exact event
duplicates, not all forms of near-duplicate or campaign leakage. Verify group
provenance yourself. Shards retain graph continuity within each capture. Batches
span storage shards, so shard size does not change the model's batching protocol.

## 3. Train a candidate

```bash
venv/bin/python train.py train \
  --data data/prepared --output training-runs/candidate-01 \
  --epochs 30 --patience 5 --device cuda \
  --batch-sizes 1,32,200 --eval-batch-size 32 \
  --min-precision 0.95 --min-recall 0.95 --max-fpr 0.001 --alpha 0.01
```

Use `--device cpu` if no compatible GPU is available. Batch sizes 1/32/200 expose
the model to different amounts of immediate history; validation uses one fixed
serving protocol. No label-based resampling breaks temporal order. Default model
sizes are memory/embedding 128, time 32, current-flow hidden 64, history 100,000.
`--node-capacity` defaults to 34,668 and must cover the largest prepared capture.
It is also the deployment node capacity. Larger dimensions/capacities consume more
RAM/VRAM. Each NumPy shard is loaded into host memory, while history is bounded.
Evaluation retains one score/label per event in host memory.

The recommended threshold maximizes validation recall subject to the requested
precision, minimum recall and false-positive-rate constraints. If no threshold meets them, the
run explicitly records `validation_target_met: false`; that candidate is not
accepted merely because training completed. A `.95` target is an empirical
validation constraint, not a promised real-world precision or confidence bound.

Outputs:

- `best.pt`: candidate selected on validation, with cleared memory.
- `run.json`, `training-history.json`: configuration, provenance and learning curve.
- `validation.json`: aggregate, per-capture and cold-batch metrics.
- `operating-point.json`: threshold, feasibility, serving batch size, hashes.
- `calibration.npz`: held-out benign raw-model scores for later conformal assessment.

The default `--selection-metric constrained_recall` first prefers epochs meeting
all three validation targets, then higher recall, then average precision, then
lower unweighted log loss. If no epoch meets the targets, average precision and
log loss select a diagnostic candidate with failed feasibility. Use
`--selection-metric average_precision` for an explicit AP-first comparison.
Targets are provisional configurable requirements, not established risk tolerances.

Calibration is protocol- and distribution-specific. The saved conformal scores
apply to the **raw model**, not `max(model_score, heuristic_score)`. Do not load
these scores into the hybrid SOAR pipeline and claim calibrated automatic blocking.
Calibrate the exact deployed composite detector separately. The scripts do not
train forecasting weights, certify MITRE classification, or automatically deploy.

## 4. Evaluate untouched data

```bash
venv/bin/python train.py evaluate \
  --data data/prepared \
  --checkpoint training-runs/candidate-01/best.pt \
  --operating-point training-runs/candidate-01/operating-point.json \
  --output training-runs/candidate-01/test-report.json \
  --batch-sizes 1,32,200 --short-window 1000 --require-targets
```

This evaluates only `test` groups, with the threshold frozen from validation.
The prepared dataset manifest must match the hash recorded in the checkpoint;
reserve external evaluation captures as `test` groups before training. Training
does not open test shards. Evaluation rejects changed manifests to prevent an
accidental reassignment of training captures into the test set.
It reports PR-AUC (average precision), ROC-AUC, precision, recall, F1, false-positive
rate, Brier score, cold-batch results, per-capture results, and stage F1 only where
stage ground truth exists. Full capture history and periodically reset short
windows are measured; reset windows round to a batch boundary. The default API
request limit is 1,000 flows. Alternative batch protocols are stress tests and
need their own validation/calibration before use.

The stored held-out binary result is for the learned model, not the existing
heuristic fusion. Do not compare it directly to historical hybrid results with
different protocols. Run the same labeled captures through every candidate and a
simple non-graph baseline, using identical splits and features. For release, also
measure end-to-end alert volume, latency, memory and false blocks on your network.

`--require-targets` writes the report and exits with code 2 if validation failed,
the serving batch protocol is missing, or the full/short serving protocols fail
aggregate or individual-capture targets. Single-class captures test the applicable
rates: benign-only captures test false positives; attack-containing captures test
precision/recall. Success means **observed binary targets met**, not production
approval, statistical certainty, or validated stage classification.

Run several seeds and tune only on training/validation; open the test once after
selecting a configuration. Repeatedly choosing models from test scores makes it
another validation set. Public benchmark accuracy alone is not an acceptance gate.

## 5. Review before selecting a checkpoint

Choose a model by independently measured performance at an acceptable false-alarm
budget, not headline accuracy. Add audited benign examples of TLS, DNS, software
updates, backups and normal administrative traffic from the target environment.
Also test held-out hosts, later time periods and attacks absent from training.
Training-data-only class weights preserve natural validation/test prevalence.

For manual candidate inference, use `ModelRuntime(candidate_path)` or
`app.py --checkpoint PATH`. For `api.py`, set `TGN_CHECKPOINT=PATH`. Send the
validated threshold and batch size explicitly to `/predict/sequence`; the API's
legacy defaults do not automatically adopt `operating-point.json`. Live detector
batching and thresholds also require deliberate configuration and matching
validation. Keep existing deployment weights until the candidate passes review.

## Software smoke test

For a new stability candidate, pass `--time-transform signed_log1p` to
`train.py train`. This applies `sign(dt) * log1p(abs(dt))` before both learned
time encoders. Newly encountered nodes can otherwise produce time deltas near
a full Unix timestamp and very large encoder gradients. The transform is saved
in `model_config` and honored by all three inference runtimes. Missing settings
retain `identity` for existing checkpoints; changing the setting requires new
training and validation, not editing a trained checkpoint's metadata.

`venv/bin/python scripts/diagnose_graph_gradients.py` reproduces the matched-seed
4096-row training probe in `reports/audit/graph-gradient-probe.json`. This small
probe measures numerical stability; it does not establish detection accuracy.

`venv/bin/python scripts/run_controlled_baselines.py` runs three feature-only MLP
comparisons: CIC2017 training only, combined CIC2017/IDS2018, and combined data
with capped attack-family weights. All use hidden width 64, seed 42, learning
rate 0.0003, batch size 1024, ten maximum epochs, patience three, and identical
capture-separated validation data. Adding data also increases updates per epoch;
this is a fixed-epoch training-policy comparison, not an equal-compute study.
These runs use `--validation-only` and never read test shards. They create fresh
`training-runs/controlled-*` directories and refuse to overwrite existing runs.

Family weights use training counts only: benign weight 1 and attack-family
weight `min(20, sqrt(largest_attack_family_count / family_count))`. They multiply
the existing binary positive-class weight and are normalized to mean 1 over
training examples. Prepared CIC2017 shards lack individual attack-family labels,
so their attacks remain one explicitly named group; no family labels are guessed.
Weighting cannot create new information or resolve indistinguishable inputs.

`venv/bin/python scripts/audit_feature_consistency.py` audits a seeded stratified
sample of training/validation features and records conflicting binary labels for
identical feature vectors. Sample counts are not estimates of overall conflict
prevalence. Arithmetic consistency is not verification of attack-label truth.

```bash
venv/bin/python -m training.smoke /tmp/cybertgn-training-example
venv/bin/python train.py prepare \
  --manifest /tmp/cybertgn-training-example/manifest.json \
  --output /tmp/cybertgn-training-example/prepared
venv/bin/python train.py train \
  --data /tmp/cybertgn-training-example/prepared \
  --output /tmp/cybertgn-training-example/run \
  --epochs 3 --device cpu --node-capacity 16 \
  --memory-dim 16 --embedding-dim 16 --time-dim 8 --direct-dim 16
venv/bin/python -m unittest discover -s tests -p test_training.py -v
```

Synthetic fixtures are intentionally easy and must never be reported as dataset
accuracy. The trainer follows the update/backpropagation/detach lifecycle in the
[official PyG TGN example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py).


## Losses, metrics and candidate selection

This model performs binary intrusion classification and optional stage
classification. It does not predict a continuous regression target. Adding an
MSE regression objective or a more complex network without a validated target
would not establish better detection.

## Training objective

The loss is `weighted_binary_cross_entropy + stage_weight * masked_stage_cross_entropy`.
Binary loss consumes logits directly through PyTorch's numerically stable
`binary_cross_entropy_with_logits`. Its positive weight defaults to
`min(20, sqrt(training_benign / training_attack))`. `--positive-weight 1` disables
this weighting; another positive value makes the tradeoff explicit. Compare such
choices using validation data. No weighting rule is universally optimal.
[PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

Stage weights use training counts only, with capped square-root inverse frequency.
Unknown stage labels (`-1`) contribute no stage loss or stage gradients. Setting
`--stage-weight 0` disables the auxiliary objective. A dataset with only benign
stage labels cannot establish useful attack-stage classification. Dataset attack
families must not be treated as verified MITRE stages without annotation.

`training-history.json` now records the binary loss, weighted stage loss over
supervised labels, combined batch objective, maximum gradient norm before clipping,
validation log loss, AP, precision, recall, F1, FPR, threshold and target feasibility.
Stage loss uses the sum of observed class weights as its epoch denominator.
The batch-averaged combined objective is not necessarily the sum of these two
epoch summaries when labeled-stage prevalence differs between batches.

Class weighting changes the learning tradeoff and can change probability
calibration. Weighted training loss and unweighted validation log loss are
different quantities. A falling loss alone does not establish better detection.

## Evaluation

Reports include TP/FP/TN/FN, precision, recall, F1, F2, specificity, false-negative
rate, FPR, balanced accuracy, MCC, AP, ROC-AUC, Brier score, clipped log loss,
ten-bin calibration error and reliability bins. The confusion matrix is ordered
`[benign, attack]`, with actual classes in rows and predicted classes in columns.
Stage confusion matrices include only supervised rows and show class support.
AP is average precision, not trapezoidal interpolation of the precision-recall curve.
Log-loss reporting clips probabilities to `[1e-15, 1-1e-15]`; training operates
directly on logits without this reporting clip.
[scikit-learn metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html).

Undefined precision/recall retain a numeric zero for compatibility and have explicit
`*_defined` flags. Missing-class balanced accuracy and false-negative rate are null.
Per-capture reporting prevents a large easy capture from hiding a failed one.

The reported 95% Wilson intervals illustrate binomial sampling uncertainty.
**They assume independent trials.** Correlated flows and shared attack campaigns
can make them too narrow. They are not a guarantee or an automatic acceptance
criterion. Even zero observed false positives has a nonzero upper uncertainty bound.
Use independently captured environments and periods to assess generalization.

## Selection and gates

Choose targets before testing. Thresholds and checkpoints are selected using
validation only; test thresholds remain frozen. The default checkpoint criterion
prefers target-feasible epochs, then recall, AP and lower log loss. An AP-only
selection mode is available for controlled comparison. Minimum recall prevents
an apparently precise model that detects almost nothing from passing.

Evaluation records empirical target checks per capture and protocol. With
`--require-targets`, unmet or missing validation/serving-protocol evidence yields
exit code 2 after preserving the report. `deployment_approved` remains false:
public-dataset metrics do not validate automatic blocking in the target network.

Use multiple seeds and a simple logistic/tree baseline with the same features and
splits before claiming the neural model is better. Select on validation, and use
untouched test data once for the selected configuration. More tuning on test data
does not strengthen the evidence. See [training commands](#retraining-cybertgn) for commands.

## Verification

Install `requirements-test.txt` for the full test suite. Independent tests compare
metrics with scikit-learn under class imbalance, ties and extreme scores, compare
weighted BCE and its gradients with analytical formulas, and verify that unknown
stages cannot change stage loss or gradients. Synthetic tests establish software
behavior only; completed CPU experiments have not met detection targets, and GPU
execution is not covered by these checks.


## IDS2018 feature-based candidates

The nine finalized local IDS2018 CSVs lack endpoint IPs. This workflow uses their
16 flow features in separate binary MLP and logistic-regression baselines. It
does not fabricate endpoints, modify CyberTGN graph weights, or deploy candidates.

The two incomplete 20 February download copies are excluded. Seven finalized
days supply training examples; 23 February and 1 March are held out for validation.
The original prepared CIC2017 Tuesday/Wednesday training rows are included.
CIC2017 Thursday remains validation and Friday remains the frozen test set.
The data manifests and checkpoint hashes identify the exact inputs.

`training/tabular.py` validates required feature names, explicitly recognizes
binary label names, rejects invalid numeric rows into indexed audit logs, saves
source/shard hashes, and computes normalization from training rows only. Original
CSVs remain unchanged. Rejected-record numbers refer to parsed CSV record order.
Records are not deduplicated solely because their 16 features match: distinct
flows can have identical statistics. Capture separation does not prove that all
cross-capture duplicates or campaign dependencies are absent.

The neural baseline has one 64-unit ReLU hidden layer. The logistic baseline has
a single linear logit. Both use weighted BCE, AdamW, gradient clipping, at most
10 epochs and patience 3. Training shards from both datasets and rows within each
training shard are shuffled; these models have no temporal state. Loss weighting
uses training counts only. Features use log1p and training-only normalization.
Timestamp, port, family label and graph identifiers are not numerical inputs.

Validation checks use 95% precision, 95% recall and FPR no greater than 0.1%, with
per-capture checks. Failed feasibility keeps the diagnostic threshold at 1.0.
`validation-at-0.5.json` exposes ordinary fixed-threshold behavior separately;
it does not override acceptance checks. Friday test evaluation uses the saved
validation threshold and precisely the same prepared CIC2017 test rows as the
graph candidate. No candidate is automatically promoted.

Run status and logs:

```bash
cat training-runs/ids2018-baselines-job/status.json
tail -f training-runs/ids2018-baselines-job/job.log
```

Preparation is independently logged in `/tmp/ids2018-tabular-prepare.log`; the
durable per-source audit is `data/ids2018-tabular/preparation-progress.json`.
Complete data metadata is `data/ids2018-tabular/dataset.json`.
Each candidate directory contains `run.json`, `history.json`, `best.pt`, validation
reports and a final `test-report.json`. Checkpoints are explicitly feature-only
and cannot be substituted directly for a graph checkpoint.

Comparing a new feature model with the first graph candidate changes both the
training data and the model architecture. It cannot establish that IDS2018 alone
caused any improvement. A subsequent same-architecture comparison is complete: see
[the remediation report](../reports/audit/training-remediation.md). Source extraction semantics,
label quality and capture dependence also remain relevant limitations.

## Closed-window forecasting and model explanations

The experimental `CyberDefensePipeline` now processes completed flows in a
consistent single-flow scoring order. It retains DNS and reorders completed flows
using the idle-time watermark (five seconds by default), preventing a newer FIN
flow from preceding an older idle flow in a closed window. The pending-flow buffer
is capped at 10,000 and reports an overflow rather than silently dropping evidence.
Call `pipeline.tick(current_timestamp)` from a live-source timer to flush idle
flows when packets stop arriving. Offline replay explicitly flushes at EOF.
Packet event timestamps must be chronological; out-of-order input is rejected.
This is a changed serving protocol, not a reuse of the earlier batch-32/200 accuracy
or calibration results. The API hybrid path remains separate.

### Train forecasts on actual future observations

Create a PCAP manifest with separate train/validation capture groups. Each source
requires `path` (relative to the manifest), `sha256`, `group`, and `split`. Only
`train` and `validation` splits are accepted; repeated file hashes/groups are
rejected. These sources need real timestamps and enough consecutive observed
15-second windows. Do not use the synthetic case studies to claim accuracy.

```bash
venv/bin/python -m training.export_forecast \
  --manifest /path/to/pcap-manifest.json \
  --checkpoint training-runs/cic2017-first-candidate/best.pt \
  --output data/forecast-windows
venv/bin/python -m training.forecast \
  --manifest data/forecast-windows/manifest.json \
  --output training-runs/forecast-candidate --epochs 50 --patience 5
```

The exporter uses the same serving pipeline with incidents/explanations disabled.
It saves mean node-memory states for active completed-flow nodes in each closed
window, plus window indices, PCAP/source hashes, and encoder identity. Capture
memory resets between sources. Empty/unobserved windows are not invented; the
last partial window is excluded. This state schema is not the proposed global
pool of attention embeddings or the separate 32-edge-feature architecture.

The trainer fits 15/30/45-second residual latent targets, normalizes using training
anchors only, and skips examples crossing missing windows or capture boundaries.
Validation measures the actual Kalman-corrected serving recurrence and persistence
at every horizon/capture. The loader rejects a candidate that does not beat
persistence at every horizon/capture or mismatches encoder hash, window size,
state dimension, schema, or flow batch size. This comparison is a research gate,
not production approval or an error-free forecasting guarantee.

```python
from pathlib import Path
from pipeline import CyberDefensePipeline

pipeline = CyberDefensePipeline(
    checkpoint_path=Path('training-runs/cic2017-first-candidate/best.pt'),
    forecast_checkpoint=Path('training-runs/forecast-candidate/best.pt'),
    dry_run_firewall=True,
)
```

Without a forecast checkpoint, closed windows return `status='untrained'` and no
predictions. With an accepted candidate, each result names its target window/end
and latent-state vector. When the next matching window closes, its observation
corrects the saved one-step prior; all three new horizons start from that corrected
state. A gap resets the prior. These are latent-state forecasts, not trained
future infiltration probabilities or MITRE timelines. No real-data forecast
checkpoint has yet been trained in this integration work.

### Bind calibration to the actual scoring protocol

`calibration.artifact.save_calibration` stores benign attack scores, optional
explicit integer stage labels/probabilities, alpha, encoder hash, serving protocol,
and source/capture provenance in a new NPZ. Use the pipeline's
`encoder_sha256` and `calibration_protocol`. Scores must have been produced with
that exact pipeline/checkpoint; training-run raw-score calibration is not a
substitute for the pipeline's fused-score protocol.

```python
from calibration.artifact import save_calibration

# Arrays and provenance below must come from an independent labeled calibration capture.
save_calibration(
    'stream-calibration.npz', pipeline.encoder_sha256, pipeline.calibration_protocol,
    benign_scores, stage_labels, stage_probabilities, alpha=0.10,
    provenance={'capture_groups': ['held-out-calibration'], 'source_sha256': source_hash},
)
```

Pass `calibration_file=Path('stream-calibration.npz'), conformal_alpha=0.10` when
constructing an otherwise identically configured pipeline. Checkpoint, protocol,
alpha, and stage-dimension mismatches fail explicitly. Calibration diagnostics
include label support and confidence ECE; these are in-sample diagnostics, not
independent evidence of detection quality or coverage under drift.

Automatic SOAR now requires all of: model score >0.85, binary conformal
confirmation, a calibrated nonempty stage set containing only Lateral Movement
and/or C2, a matching predicted stage, and `stage_validation.targets_met=True` in
the trusted graph checkpoint. Current candidates have no such stage-validation
evidence (the CIC2017 run records zero supervised stage rows). Do not add this
flag manually to bypass the gate: it must follow independently documented stage
validation for the serving protocol. Ambiguous sets containing benign/unknown
stages abstain. Victim quarantine requires a singleton Lateral Movement set.
Firewall operations still default to dry-run.

### Explain actual model scores

The pipeline captures pre-update memory and history for each scored flow.
`explainability/model_explainer.py` produces:

- Permutation Shapley estimates for the 16 current-flow features, using the v2
  training-feature mean as a single reference. It reports base score, prediction,
  per-feature Monte Carlo standard errors, and an additivity residual. Correlations
  between features are not preserved by these reference interventions.
- PyG GNNExplainer masks optimized on historical edges for the predicted class,
  with the highest-weight edges mapped back to IPs. These explain the frozen
  graph calculation, not the causal contribution of past events to memory.

These methods explain the graph model, not heuristic fusion. A legacy model can
correctly report zero current-feature contribution. A cold-start graph has no
historical edges to explain. The pipeline has an explanation-count budget per
`process_flows` call and an edge/optimization budget; skipped or failed explanations
report their status without substituting heuristic weights or dropping forensic
logging. The dashboard does not yet render these new explanation objects.

Implementation follows the [PyG GNNExplainer interface](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.GNNExplainer.html).
