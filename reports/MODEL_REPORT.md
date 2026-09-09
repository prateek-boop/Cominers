# CyberTGN evaluation report

> Historical report. After cleanup on September 9, 2026, the retained checkpoint
> is `checkpoints/tgn_best.pt`. Training/evaluation scripts, datasets, archives,
> and other checkpoints were removed. References below describe the original
> evaluation environment, not files currently available for rerunning it.

The best checkpoint shows useful binary detection on long validation chunks,
but **this model is not ready for reliable live deployment**. Short sequences
perform poorly, one-event processing is much worse, and attack-stage
classification fails on the evaluated attack classes.

## Full validation comparison

Evaluated **1,392,398 flows**: 8,663 attacks and 1,383,735
benign events. Both checkpoints used the same 34 validation chunks, threshold
0.5, batches of 200, and empty memory/neighbors at the start of each chunk.

| Metric | Best checkpoint (epoch index 5) | Final checkpoint (epoch index 11) |
|---|---:|---:|
| Accuracy | 99.79% | 98.52% |
| Attack precision | 76.35% | 28.47% |
| Attack recall | 96.20% | 91.02% |
| Attack F1 | 0.8513 | 0.4337 |
| ROC AUC | 0.9987 | 0.9934 |
| Average precision | 0.7164 | 0.5695 |
| False alarms | 2,582 | 19,810 |
| Missed attacks | 329 | 778 |

Predicting every event as benign already gives **99.38%
accuracy**, so accuracy alone overstates usefulness. The best checkpoint caught
8,334 attacks. Its false-positive rate was
0.187%; roughly 24% of its alerts were false alarms.
Use `tgn_best.pt`, not `tgn_final.pt`, for the supplied service.

The checkpoint records a historical validation F1 of 0.9562. This fresh run
measured 0.8513 under the explicitly different cold-chunk protocol. Original
validation retained shuffled training history. The original random training
order and exact run arguments were not saved, so this is not an exact replay
of that historical score.

## Short sequences and processing mode

A matched prefix sample uses the first 1,000 flows of each validation chunk:
**34,000 flows, including 203 attacks**. These results cannot be compared as
if they were the full dataset above; their purpose is to compare processing
modes on identical events and expose cold-start behavior.

| Metric | Batches of 200 | One event at a time |
|---|---:|---:|
| Precision | 14.58% | 1.29% |
| Recall | 84.73% | 12.81% |
| F1 | 0.2487 | 0.0234 |
| False alarms | 1,008 | 1,992 |
| Missed attacks | 31 | 177 |

The API defaults to batches of 200 to match training. Batch size 1 is an
explicit experimental option. **Even batches of 200 perform poorly on these
short prefixes.** The full-run F1 must not be advertised as short-request API
accuracy. Retraining and validation for the intended serving pattern are
needed. Request length, batch size, node history and dataset composition
all affect results.

The model consumes current features only after scoring a batch. Its first
batch has no historical evidence and produces cold-start scores. A standalone
`/predict` call therefore cannot judge a flow from its own features.

## Stage classification

The best checkpoint's stage macro F1 is **0.2492**, calculated
over labels observed or predicted (not all 14 outputs). Attack classes 1 and 2
have **zero precision, recall and F1** in this full validation run. Almost all
correct stage predictions are benign. Do not rely on the stage output for
attack attribution. The supplied mapper defines IDs 0–7, although the model
has 14 outputs; the API labels IDs 8–13 as unmapped.

## Evidence and limits

- The existing extracted files matched archive CRCs; no re-extraction was needed.
- The split was reconstructed from archived training defaults: chunks >= 59,
  first 80% of attack-containing and benign-only chunks for training.
  There is no train/validation **file** overlap; row-level duplicate leakage
  was not checked.
- This validation split was used for checkpoint selection. It is **not an
  untouched independent test set**. No external-generalization claim is made.
- The audit covered 8,031,325 eligible flows: 58,674 all-zero feature rows,
  including 5,473 validation rows; no non-finite feature values were found.
- Preprocessing compacts missing feature columns instead of retaining their
  fixed positions, rounds timestamps through float32, and maps unknown attack
  labels to class 1. Dataset source manifests and the original IP map are absent.
- Per-chunk metrics, exact file lists, checkpoint hashes and protocol details
  are saved in the accompanying JSON reports.

## Delivered software and checks

`app.py` runs the interactive FastAPI application or predicts from a JSON file.
`api.py` exposes the same trained model for other programs. `model_runtime.py`
loads checkpoint weights strictly, resets graph state between callers, uses
collision-free IP IDs and preserves predict-then-update behavior.
`evaluate_model.py` reproduces the measurements.

Six tests passed against the real checkpoint and in-process HTTP API:
checkpoint/repeatability, state isolation, history-dependent feature effects,
feature transformation, default batch boundaries, validation and HTTP schema
checks. CLI JSON prediction also passed. HTTP tests required execution outside
the sandbox because the sandbox stalled asynchronous worker notification.
No public deployment was performed.

See [MODEL_USAGE.md](../MODEL_USAGE.md) for install/run commands, request format,
feature order, sharing instructions and a Python client.
