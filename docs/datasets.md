# Datasets for CyberTGN retraining

No dataset combination guarantees “best results.” For this project's existing
16-feature graph interface, start with corrected CIC flows and use additional
environments to measure generalization. The choices below are recommendations
based on task and schema fit, not a benchmark ranking. Sources checked 2026-09-09.

| Dataset | Recommended role | Compatibility and source |
|---|---|---|
| **Corrected CIC-IDS2017** | First pipeline-training dataset and reproducible baseline | Prefer the corrected labels/extractor; retain IPs and timestamps. [DistriNet corrected datasets](https://intrusion-detection.distrinet-research.be/CNS2022/index.html), [original UNB dataset](https://www.unb.ca/cic/datasets/ids-2017.html). |
| **Corrected CSE-CIC-IDS2018** | Larger enterprise-style training/evaluation captures | Use a separate environment/capture split; validate every feature mapping. [DistriNet downloads](https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html), [UNB](https://www.unb.ca/cic/datasets/ids-2018.html), [AWS public registry](https://registry.opendata.aws/cse-cic-ids2018/). |
| **UNSW-NB15** | Broader attack-family and cross-environment evaluation | Nine attack types; raw PCAP and Argus/Bro/CSV exports exist. Its tabular schema needs re-extraction or reviewed adaptation. [Official UNSW source](https://research.unsw.edu.au/projects/unsw-nb15-dataset). |
| **ToN-IoT (network subset)** | IoT/IIoT and mixed-environment evaluation | Includes PCAP and Zeek network data plus separate telemetry/OS modalities. Use the network subset and its ground truth. [Official UNSW source](https://research.unsw.edu.au/projects/toniot-datasets). |
| **IoT-23** | Malware/C2 behavior and independent IoT captures | Provides labeled connection logs and raw captures. Zeek labels/features need careful flow matching and re-extraction. [Official Stratosphere source](https://www.stratosphereips.org/datasets-iot23). |
| **CICIoT2023** | IoT flood, reconnaissance and device-oriented coverage | 33 attacks in seven categories across 105 devices; supplied feature tables differ from this model's 16 features. [Official UNB source](https://www.unb.ca/cic/datasets/iotdataset-2023.html). |
| **Audited traffic from the deployment environment** | Final calibration and prospective shadow-mode testing | Capture with authorization; independently label representative benign activity and controlled attack exercises. This addresses environment mismatch that public lab datasets cannot resolve. |

## Recommended sequence

1. Validate the importer and initial model on corrected CIC-IDS2017 captures.
2. Develop additional candidates with corrected CSE-CIC-IDS2018; preserve an entire
   source/environment for external testing instead of mixing all sources blindly.
3. Add ToN-IoT/IoT-23/CICIoT2023 only when IoT or industrial traffic matches the
   intended environment. UNSW-NB15 is useful as another independent environment.
4. Calibrate on audited target-network traffic and test on a later untouched period.
5. Select by recall at the required false-positive/alert budget, stage quality where
   ground truth exists, and serving latency—not aggregate accuracy alone.

## Data-quality requirements

The authors of the corrected CIC datasets documented errors in attack generation,
feature extraction and labeling in the original releases. Their corrected data and
labeling logic are preferable starting points, subject to your own audit.
[Author's study and tools](https://intrusion-detection.distrinet-research.be/CNS2022/index.html).
Their “Attempted” labels require an explicit preprocessing decision; follow the
[authors' guidance](https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html)
and record that decision rather than treating “Attempted” as a new attack family.

- Download from the linked custodians; record release/version and file hashes.
- Keep capture identities, endpoints, chronology and ground-truth provenance.
- Feature-only ML CSVs may omit IPs. Use labeled flow exports/raw captures with
  endpoints for a graph model. Never fabricate graph relationships.
- Re-extract all sources with the **same serving feature implementation** when
  possible, then join labels using endpoints, ports, protocol and time intervals.
  Reject ambiguous joins or mixed-label flows. A dataset-specific label joiner is
  not included; the importer accepts already-labeled canonical/CIC CSVs.
- Do not zero-fill missing CIC features in a Zeek/Argus/CICIoT export. Those schemas
  measure different quantities, even when names sound similar.
- Hold out captures/campaigns and later periods. Random row splitting can share
  nearly identical flows, host history or attack campaigns between train and test.
- Never equate an attack-family label with a confirmed MITRE stage automatically.
  Use `stage: null` until an authoritative mapping/annotation exists.
- Keep evaluation prevalence realistic. Oversampled attack-heavy test sets produce
  misleading estimates of precision and false-block volume.

Check each custodian's usage terms before redistribution or commercial use. For
example, ToN-IoT explicitly distinguishes academic use and commercial permission.
[ToN-IoT terms](https://research.unsw.edu.au/projects/toniot-datasets).

Use [training guide](training.md) for commands, schemas, split design and candidate
acceptance. Do not replace the deployed checkpoint based on synthetic smoke tests.
