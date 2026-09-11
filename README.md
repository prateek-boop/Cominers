# CyberTGN — network intrusion detection research prototype

Status reviewed: **11 September 2026**. CyberTGN captures/replays traffic, builds
flows, scores them with heuristics and a temporal graph model, and demonstrates
response and forensic logging. **It does not yet meet the proposed autonomous
cyber-defense specification or its detection-quality requirements.**

The eight-phase assessment below distinguishes working components from integrated,
trained, and operationally validated behavior. Code exists for all eight phases;
**0 of 8 phases is fully verified against every requirement in the supplied
specification.** That is a requirements-verification count, not “0% implemented.”
A single completion percentage would hide major differences in importance and
readiness, so none is claimed.

- [Attack coverage and priorities](attack.md)
- [Where we are failing right now](attack.md#where-we-are-failing-right-now)
- [Training, losses, metrics, and experiments](docs/training.md)
- [Dataset sources and schema requirements](docs/datasets.md)
- [Latest measured results](reports/audit/training-remediation.md)
- [Historical checkpoint evaluation](docs/model-history.md)
- [Latest software integration checks](reports/audit/integration-verification.json)

This README is a manually reviewed project snapshot. Update the status, metrics,
and evidence links after new experiments or implementation changes; they do not
update automatically.

## What actually runs

Two paths currently coexist; the dashboard does not expose the entire experimental
pipeline automatically.

```text
api.py: live Scapy / uploaded PCAP
  -> FlowEngine -> FastHeuristicEngine + StreamingModelRuntime
  -> alerts / simulated mitigation -> FastAPI dashboard + WebSocket

pipeline.py: Scapy / PCAP replay
  -> NoiseFilter + AdaptiveGraphPruner -> FlowAggregator -> GraphFormatter
  -> StatefulCyberTGN + binary conformal check
  -> closed-window forecast / stage conformal set / model-based explanation
  -> SOAR hooks -> available raw PCAP evidence -> SHA-256 / Merkle / local ledger

train.py: labeled endpoint-bearing CSV -> prepared temporal shards -> graph candidate
training.tabular: feature-only IDS2018 + CIC2017 -> separate MLP/logistic candidates
```

## Alignment with the proposed eight phases

“Partial” means some working implementation exists but the complete requested
behavior is missing or unverified. “Prototype” means the central scientific or
operational claim has not been established.

| Phase | Current implementation | Gaps against the requested specification | Assessment |
|---|---|---|---|
| **1. Capture, pruning, dual-branch features** | Scapy live capture/PCAP replay; noise filters; bidirectional flows; entropy and handshake helpers. [Dual-branch helpers](features/dual_branch.py) can construct 32 features. | No production Zeek/eBPF ingestion adapter or CTU-13 import workflow. Model-serving paths still consume **16 flow features**, not the combined 32-vector. The pipeline now retains DNS; an audited internal-DNS allowlist is still needed if selective pruning is desired. Pruning uses fixed rules rather than learned/adaptive capacity control. Capture loss and million-packet throughput are unverified. | Partial |
| **2. Dynamic temporal graph** | [GraphFormatter](graph/graph_formatter.py) maps IPs to nodes and formats temporal interactions. Completed flows now feed closed 15-second memory windows with an idle-time reorder buffer. A separate [snapshot builder](graph/temporal_windows.py) exposes 16-dimensional node and 32-dimensional edge feature tensors. | Snapshot builder is not wired into the trained serving path or an automatic 15-second scheduler. Nodes are keyed by IP; roles are guessed from ports/address patterns, not separate verified IP/service identities. Some node metrics are proxies: packet volume is cumulative, not an actual window delta. | Partial |
| **3. TGN encoding and memory** | [TGN memory](models/memory.py) uses GRU updates; [graph embedding](models/tgn.py) uses attention; streaming state and bounded edge history exist. Stable signed-log time encoding is available for new training. | Message generation uses PyG `IdentityMessage`, not the specified learned message MLP. Learned model input remains 16-dimensional. Forecast pooling uses active completed-flow node memory per window, not the specified global mean of attention embeddings. Buffer/generation resets and short-sequence behavior need operational validation. | Partial |
| **4. Delta forecasting and Kalman correction** | [Closed-window runtime](forecasting/window_forecaster.py) matches priors to the next observed window, restarts from corrected states, and resets on gaps. [PCAP exporter](training/export_forecast.py) and [trainer](training/forecast.py) validate the full correction loop against persistence per horizon/capture. | No real-data forecast candidate has been trained or accepted. Runtime abstains without a compatible checkpoint that passes validation. Latent-state forecasts are not infiltration probability timelines. No zero-error guarantee. | Partial |
| **5. Binary and MITRE stage heads** | Binary and stage heads exist; v2 adds current-flow features. Training masks unknown stage labels. | Heads classify observed flow context, not trained future-state timelines. Dataset family labels do not establish MITRE stages. Attack-stage ground truth is missing for the new candidates; historical stage quality is weak. No supported family-specific accuracy or actor attribution exists. | Partial |
| **6. Split conformal calibration** | Binary calibration and [multiclass prediction sets](calibration/conformal.py), finite-sample quantiles, and [ECE computation](calibration/ece.py) exist. | Calibration artifacts can now be loaded with checkpoint/protocol/alpha checks; representative stage calibration has not been supplied. Binary benign-score confirmation is separate from multiclass stage coverage. No guarantee under arbitrary drift/OOD traffic; coverage requires suitable calibration/exchangeability. Pipeline defaults to alpha 0.05, whereas the requested example uses 0.10. No live ECE dashboard is wired. | Partial |
| **7. GNNExplainer and Edge SHAP** | The pipeline now uses [permutation Shapley estimates and PyG GNNExplainer](explainability/model_explainer.py) on frozen pre-update model state. | Explanations cover current-flow features and historical edges conditional on fixed memory; they do not explain memory formation or heuristic fusion. Sampling/optimization are approximate, have budgets, and need operational fidelity/latency assessment. The dashboard view is still missing. | Partial |
| **8. SOAR, forensics, Merkle verification** | Firewall/quarantine/decoy command hooks; dry-run default; raw-packet extraction when evidence exists; SHA-256 hashing, Merkle proofs, and a local hash-chain ledger. | eBPF/VLAN tools and firewall chains require external provisioning. Pipeline SOAR now requires model score >0.85, binary confirmation, explicit stage-validation evidence, and a calibrated nonempty set containing only Lateral Movement/C2. Current checkpoints lack the required stage evidence, so this gate abstains. Commit is per incident batch, not a 60-second scheduler. No Hyperledger/testnet anchoring, immutable PCAP store, or one-click dashboard proof. Real enforcement, recovery, latency, and throughput are unverified. | Partial |

### Four-layer view

Telemetry capture is functional in local workflows. Feature/threat modeling has
working flow statistics and an experimental classifier. Automated response has
command paths but defaults to simulation and lacks validated detection gates.
Forensics offers local evidence and tamper detection, not externally anchored
immutability. These are useful building blocks, not a completed autonomous system.

## Measured detection quality

The latest matched feature-model experiments completed seven epochs each, selecting
epoch four. Same 64-unit MLP, seed 42, validation captures, and stopping rule;
adding data increases optimizer updates per epoch. These are single-seed,
validation-only comparisons, not equal-compute or final deployment benchmarks.

| Training configuration | Average precision | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | False-positive rate at 0.5 |
|---|---:|---:|---:|---:|---:|
| CIC2017 only | 0.054645 | 9.91% | 0.46% | 0.008814 | 0.238% |
| CIC2017 + IDS2018 | 0.101922 | 9.03% | 26.82% | 0.135103 | 15.319% |
| Combined + family weighting | 0.103567 | 11.01% | 35.63% | 0.168266 | 16.315% |

**None met precision ≥95%, recall ≥95%, and false-positive rate ≤0.1%.**
The threshold 0.5 is a diagnostic, not an approved operating point. Average
precision is a ranking metric, not classification accuracy. These results concern
separate feature-only candidates, not the deployed legacy graph checkpoint or
heuristic rules. [Machine-readable results](reports/audit/controlled-baselines.json)
and [logs](reports/audit/controlled-baselines.log) preserve the evidence.

The sampled data audit found 182 distinct input vectors with conflicting binary
labels among 11,618 distinct sampled vectors. It does not establish a population
error rate, but shows that the selected features can be ambiguous. The graph-time
probe reduced maximum gradient norm from 11.8 million to 7.8; that measures
numerical stability, not an accuracy improvement. The latest cleanup regression run passed **96 tests**; see the [test log](reports/audit/code-cleanup-tests.log).
One test for the removed rule-based explainers was retired; model-explanation tests remain.
Software tests do not validate real-world detection.

## Alignment with the named case studies

SolarWinds/SUNBURST, Salt Typhoon, Volt Typhoon, and Cisco exploit examples in
[casestudies](casestudies/) are synthetic demonstrations. They do not reproduce or
validate detection of those campaigns. The attack inventory explains the missing
telemetry and evaluation needed for each.

DNS is now retained by the experimental pipeline, but this alone does not establish
DNS-based C2 detection. Network flow statistics cannot establish malicious use of
valid credentials without identity, asset, and configuration context.
Port 8443 or an IP ending in `.250` is not proof
of a lawful-intercept system. A signed update does not establish safe behavior,
but this project also does not inspect vendor build provenance or signed binaries.

There is a factual correction to the supplied CVE context: **CVE-2019-1653 concerns
Cisco RV320/RV325 information disclosure; CVE-2019-1654 concerns Cisco Aironet
AP development-shell access, not an RV320/RV325 command-injection vulnerability.**
See the [Cisco RV advisory](https://sec.cloudapps.cisco.com/security/center/content/CiscoSecurityAdvisory/cisco-sa-20190123-rv-info)
and [NVD Aironet record](https://nvd.nist.gov/vuln/detail/CVE-2019-1654).

## Work required before claiming full alignment

1. Collect independently labeled rare attacks,
   representative benign traffic, endpoint/authentication logs, and router changes.
   Join labels to flows with provenance; do not derive ground truth from a port.
2. Connect one consistent packet/flow feature schema and 15-second graph builder
   to both training and serving. Train compatible weights before changing input
   dimensions. Benchmark memory, packet loss, processing delay, and reset behavior.
3. Train and evaluate binary and stage detection per attack family, capture, and
   short-sequence length, with multiple seeds and an untouched final evaluation.
4. Use the new exporter/trainer to fit delta forecasts on real future-window targets.
   The correction loop is implemented; measure every horizon on independent captures.
5. Supply independent stage calibration and validation evidence for the implemented
   SOAR gate. Measure drift and false alarms rather than treating a sigmoid as confidence.
6. Connect the implemented model explanations to the requested dashboard views; validate live
   enforcement/reversal, evidence retention, timed batching, and external anchoring.

Neither bounded forecasting nor a Merkle tree implies “error-free” prediction or
“zero latency.” Both accuracy and system overhead require measurement.

## Integrated model workflow

The experimental pipeline now uses fixed single-flow scoring, a bounded reorder
buffer, and closed 15-second completed-flow windows. This changes its serving
protocol: completed flows wait for the idle-time watermark (default five seconds),
and processing can wait longer until the next packet or timer tick. Offline replay
flushes remaining flows explicitly. An idle live source needs a timer that calls
`pipeline.tick(current_timestamp)` to flush expired flows and advance windows.
API hybrid detection remains a separate path. Earlier accuracy/calibration results
do not validate this changed pipeline. See [workflow instructions](docs/training.md#closed-window-forecasting-and-model-explanations).

## Run locally

Use Python 3.12 and install dependencies if setting up a fresh environment:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements-inference.txt
venv/bin/python api.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard` or `http://127.0.0.1:8000/docs`.
Live capture requires suitable interface/OS permissions and visibility from a
mirror port/TAP where needed. The service has no built-in authentication and
binds to loopback by default. Firewall behavior defaults to simulation. PCAP
uploads are capped at 64 MiB and use independent analysis state.

```bash
# JSON inference through the legacy-compatible entry point
venv/bin/python app.py --input examples/flows.json --output /tmp/predictions.json
# Experimental architecture demo (synthetic, temporary artifacts by default)
venv/bin/python demo.py
# Existing software checks
venv/bin/python -m pip install -r requirements-test.txt
venv/bin/python -m unittest discover -s tests -v
```

Primary routes: `/health`, `/model`, `/predict`, `/predict/sequence`,
`/analyze/pcap`, `/sniff/start`, `/sniff/stop`, `/sniff/status`, `/alerts`,
`/mitigation/unblock`, `/model/reset`, `/ws/traffic`.

For JSON inference, send ascending Unix timestamps, IP endpoints, and exactly
16 finite features in the order returned by `/model`. The legacy checkpoint's
first batch has no historical evidence and does not use current-flow features;
v2 candidates address that architectural limitation. Sequence requests reset
memory; live streaming has persistent state. Scores and stage outputs are not
validated breach probabilities or verified MITRE annotations.

`checkpoints/tgn_best.pt` remains the existing checkpoint. New graph candidates
need an explicit trusted checkpoint path; feature-only MLP/logistic checkpoints
are not graph-compatible replacements. Consult the training guide before changing
serving thresholds, batch size, feature encoding, or weights.

## Repository organization

`attack.md` and this README are the two root documentation entry points.
`docs/` retains consolidated training/data guidance, UI design context, and historical evaluation.
`reports/audit/` retains current findings and machine-readable evidence.
Runtime packages, scripts, tests, raw/prepared datasets, checkpoints, candidate
runs, forensic PCAPs, and ledger records are retained. Cleanup removes superseded
documentation, stale test logs, failed report renders, and Python bytecode caches;
[cleanup inventory](docs/cleanup.json) records the exact paths and recovery archive.
