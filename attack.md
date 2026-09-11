# Attack coverage and remaining work

Reviewed **11 September 2026** against the implementation and saved experiments.
**No attack family currently has sufficient evidence to claim production-reliable
ML detection.** The neural heads output a binary attack score and a stage score,
not a validated attack-family or threat-actor identity. An attack's presence in
training does not establish that the model can detect it.

## Where we are failing right now

This is a manually maintained status snapshot, not an automatically updating
monitor. Update it after new training, evaluation, or integration changes.

| Area | Current evidence / limitation | What closes the gap |
|---|---|---|
| **Detection quality — measured failure** | The weighted combined-data MLP achieved **11.01% precision, 35.63% recall, F1 0.1683, and 16.32% false-positive rate** on validation at threshold 0.5. No tested candidate met the required precision ≥95%, recall ≥95%, FPR ≤0.1% operating point. | Improve data/features and demonstrate the targets on independent captures with a frozen validation-selected threshold. |
| **Attack-specific reliability — unverified** | Aggregate scores do not tell us exactly which individual attacks are detected reliably. The family inventory below records data support, not measured family accuracy. | Produce per-family support, confusion counts, precision/recall/F1, and benign false-alarm results. |
| **Rare attacks — insufficient coverage** | IDS2018 training has only **249 web-brute-force, 79 XSS, and 34 SQL-injection** rows. | More independent labeled examples and application/authentication context; weighting alone did not solve the gap. |
| **MITRE stages — missing supervision** | The CIC2017 graph candidate records **zero supervised training rows for all eight stages**. A displayed stage is not validated ground truth. | Obtain independently annotated stage labels, retrain, and validate stages under the serving protocol. |
| **Forecasting — not trained on real data yet** | Closed-window export, training, and Kalman integration exist, but no real-data forecast checkpoint has been trained and accepted. | Train on actual capture windows; beat persistence at every horizon/capture and assess prospective performance. |
| **Graph/dashboard integration — incomplete** | The trained interface still uses 16 flow features. The 32-edge-feature/role-node path and new pipeline outputs are not integrated into the API dashboard. | Complete compatible training/serving integration and dashboard wiring, then verify end-to-end behavior. |
| **Calibration and automatic blocking — not approved** | Artifact checks and stricter stage-set gates exist, but representative stage calibration/validation is missing. The changed single-flow/window protocol also needs fresh evaluation. | Supply matching independent calibration and stage-validation evidence, meet detection targets, and verify real enforcement/reversal. |

The numerical results above describe a **feature-only MLP**, not the deployed
graph checkpoint, heuristic rules, or newly integrated pipeline. They must not be
presented as current performance measurements for those other paths.

Recent software fixes are **completed but not accuracy evidence**: DNS retention,
causal window ordering, matching Kalman corrections, checkpoint-bound calibration,
and model-based Shapley/GNNExplainer methods. The full suite passed 96 tests after obsolete rule-explainer cleanup; this
does not establish reliable attack detection. Named campaign/CVE detection remains
unvalidated as detailed below.

Evidence: [weighted validation diagnostics](training-runs/controlled-family/validation-at-0.5.json),
[graph training support](training-runs/cic2017-first-candidate/run.json),
[controlled comparison](reports/audit/controlled-baselines.json), and
[latest integration verification](reports/audit/integration-verification.json).

## Patterns the current application can flag

These are implemented **heuristic rules**, separate from the trained model.
They run on completed flows and accumulated history; “can flag” means the rule
exists, not that real-world precision/recall has been established. Rule scores
such as 0.96 are fixed heuristics, not measured confidence.

| Pattern / output name | Current rule or signal | Work still needed |
|---|---|---|
| Port scanning — `PORT_SCAN` | Source reaches at least 12 distinct destination ports within 10 seconds by default. | Slow/distributed scans, host sweeps, authorized scanners, per-family validation. |
| TCP SYN flooding — `TCP_SYN_FLOOD` | Repeated small unidirectional TCP/SYN-like flows without responses. | Asymmetric capture, handshake failures, legitimate bursts, measured detection delay/FPR. |
| UDP flooding — `UDP_FLOOD` | At least 10 forward packets, no backward packets, and ≥100 packets/sec. | Benign one-way UDP and reflection/amplification validation. |
| Slowloris-like DoS — `SLOWLORIS_DOS` | TCP duration >15 seconds, low throughput, small packets, long forward IAT. | Real Slowloris/SlowHTTPTest captures and benign idle connections; flow timeout interaction. |
| Volumetric DoS/DDoS-like traffic — `VOLUMETRIC_DDoS` | ≥20 forward packets plus >10,000 packets/sec or >50 MB/sec. | Legitimate transfers, multi-source aggregation, individual DDoS variants. |
| Suspicious entropy / SYN imbalance in the experimental pipeline | High entropy or abnormal handshake ratios can raise a score and heuristic stage. | Encrypted benign traffic creates false alarms; these signals do not establish C2, exploitation, or an ATT&CK stage. |

Implementation: [hybrid rules](engine/hybrid_detector.py) and
[stateful pipeline scoring](engine/stateful_tgn.py). The first five labels belong
to the API's hybrid path; the last row describes the separate pipeline path.
A `GRAPH_ANOMALY_*` label is a generic model alert, not a proven attack identity.

## Attack families in the available training/evaluation data

All rows below still need independent family-level evaluation. Current aggregate
metrics must not be copied into these rows as attack-specific scores.

| Attack family | Data support in the completed experiments | What we must work on |
|---|---|---|
| FTP password brute force / FTP-Patator | CIC2017 graph training; IDS2018 has 193,354 accepted training rows. | Distinguish legitimate retries; auth-log correlation and held-out precision/recall. |
| SSH password brute force / SSH-Patator | CIC2017 graph training; IDS2018 has 187,589 training rows. | Low-rate/distributed attempts, valid-account use, independent hosts. |
| DoS Hulk | CIC2017 training; IDS2018 has 461,912 training rows. | Family-specific recall and benign-load false alarms. |
| DoS GoldenEye | CIC2017 training; IDS2018 has 41,508 training rows. | Different tools, environments, and legitimate HTTP bursts. |
| DoS Slowloris | CIC2017 training; IDS2018 has 10,990 training rows. | Real attack validation of both heuristic and neural paths. |
| DoS SlowHTTPTest | CIC2017 training; IDS2018 has 139,890 training rows. | Rare modes, labeling review, and timeout/sequence behavior. |
| Heartbleed | Only 11 accepted canonical CIC2017 training rows before prepared-data deduplication. | More independently verified examples and protocol context; insufficient evidence for detection claims. |
| Web brute force | IDS2018 training: **249** rows; CIC2017 Thursday is validation. | More independent examples and application/authentication context. |
| Cross-site scripting (XSS) | IDS2018 training: **79** rows; CIC2017 Thursday is validation. | HTTP/application telemetry, labeled benign lookalikes, reliable family detection. |
| SQL injection | IDS2018 training: **34** rows; CIC2017 Thursday is validation. | Richer application features and independently labeled attacks. Weighting 34 rows cannot replace data. |
| Infiltration | IDS2018 training: **68,236** rows; IDS2018 March 1 and CIC2017 Thursday are validation. | Label review, topology/identity context, severe cross-capture weakness. The label alone does not prove lateral movement. |
| Bot / botnet activity | IDS2018 training: **286,191** rows; CIC2017 Bot is in the held-out Friday split. | Validated beacon/sequence features, benign periodic traffic, C2 ground truth. “Bot” is not verified Ares or CTU-13 coverage. |
| DDoS HOIC | IDS2018 training: **686,012** rows. | Independent campaigns and benign-load controls; dominant training count is not reliability. |
| DDoS LOIC-UDP | IDS2018 training: **1,730** rows. | More capture diversity and variant-specific validation. |
| DDoS LOIC-HTTP | The two local February 20 IDS2018 download fragments were excluded as incomplete. CIC2017 generic DDoS is held out. | Complete verified source data; do not equate generic DDoS labels with this variant. |
| PortScan | CIC2017 Friday held out; absent from the CIC2017 training families. | Add independent labeled training captures without recycling the test split; validate the existing scan rule separately. |
| CTU-13 Neris scanning / IRC C2 | No completed CTU-13 preparation/training experiment. | Obtain/verify captures and labels, implement compatible extraction, preserve endpoints and sequence timing. |

IDS2018 counts refer to accepted prepared training rows. The graph candidate was
trained on CIC2017; combined IDS2018 experiments trained **separate feature-only
models** because the finalized local IDS2018 CSVs omit IP endpoints. They did not
add these families to the deployed graph checkpoint.

Sources: [coverage audit](reports/audit/cic2017-coverage-diagnosis.json),
[feature/family audit](reports/audit/feature-consistency.json), and
[combined-run provenance](training-runs/controlled-combined/run.json).

## Named campaigns and vulnerabilities: not validated detections

| Scenario | Current capability and required work |
|---|---|
| **SolarWinds / SUNBURST** | Synthetic example only. The pipeline now retains DNS, but there is no validated DNS-beacon detector, build-integrity telemetry, or endpoint/process correlation. Test independently labeled sequences and benign vendor updates. |
| **Salt Typhoon / telecom management abuse** | Synthetic router/management traffic only. Need actual asset roles, configuration changes, authentication/AAA events, lawful-intercept access context where applicable, and authorized labeled exercises. Port/IP heuristics cannot identify telecom compromise. |
| **Volt Typhoon / living-off-the-land / router relays** | Synthetic example only. Need identity, administrative-tool, device, and network-sequence correlation. No actor attribution or validated proxy-relay detection. |
| **CVE-2019-1653 — RV320/RV325 information disclosure** | No validated CVE-specific detector. Need device/software inventory, appropriate management HTTP telemetry, and independently verified labeled traffic. |
| **CVE-2019-1654 — Aironet development-shell access** | This is **not** an RV320/RV325 command-injection CVE. Requires device-local authentication/CLI context; the flow model has no validated detector for it. |
| **SMB/RDP lateral movement, exfiltration, novel exploits** | Stage outputs and synthetic examples are not ground truth. Need independently labeled multi-host sequences, identity/application context, and explicit unknown-attack evaluation. |

The CVE distinction is supported by [Cisco's RV320/RV325 advisory](https://sec.cloudapps.cisco.com/security/center/content/CiscoSecurityAdvisory/cisco-sa-20190123-rv-info)
and [NVD's Aironet record](https://nvd.nist.gov/vuln/detail/CVE-2019-1654).
For the campaign requirements, use authoritative defensive context from
[CISA's communications-infrastructure guidance](https://www.cisa.gov/resources-tools/resources/enhanced-visibility-and-hardening-guidance-communications-infrastructure)
and [Volt Typhoon advisory](https://www.cisa.gov/sites/default/files/2024-02/aa24-038a-jcsa-prc-state-sponsored-actors-compromise-us-critical-infrastructure_1.pdf).
The missing-capability assessments above come from this repository, not from those advisories.

## Priority order and acceptance criteria

1. **Infiltration, web brute force, XSS, SQL injection:** weakest target coverage;
   improve independently labeled examples and features before increasing epochs.
2. **C2/DNS and lateral movement:** retain relevant telemetry, verify stage labels,
   and add identity/asset context. Do not label all encrypted traffic malicious.
3. **Floods/scans/brute force:** measure the existing rules and models on benign
   lookalikes, new hosts, and different capture periods.
4. **Named campaigns/CVEs and forecasting:** establish appropriate telemetry and
   a dedicated evaluation; synthetic scenario success is insufficient.

Before promoting any family to “validated,” record checkpoint and extractor hashes,
independent capture identities, support counts, TP/FP/TN/FN, precision, recall, F1,
false-positive rate, threshold, uncertainty, and serving batch/window settings.
Select thresholds on validation; use fresh final evaluation for the selected
configuration. Confirm overall and per-capture targets (precision ≥95%, recall
≥95%, FPR ≤0.1%) and a suitable family-level coverage requirement. The latest
candidates fail the overall targets; see the [results](reports/audit/training-remediation.md).
