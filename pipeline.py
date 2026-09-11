"""End-to-End Cyber Defense Orchestrator.

Aligns and coordinates all 8 Phases + 4 architectural layers + AI/ML Engine:
1. Telemetry Capture Layer (Scapy Sniffer / Ring Buffer / Adaptive Graph Pruner)
2. Feature Extraction & Threat Modeling Engine (Noise Filter, Entropy, Handshake, Port Roles, Flow Aggregator)
3. Dynamic Temporal Graph Construction (15s snapshot, node roles, edge tensors)
4. AI/ML Engine (Stateful CyberTGN + Residual Delta Forecast + Kalman Re-Anchoring)
5. Multi-Task MITRE ATT&CK Stage Prediction & Dataset Ground Truth Mapping
6. True Statistical Calibration (Split Conformal Bound with Coverage Guarantee + ECE)
7. Explainability Layer (Subgraph Explainer + Edge SHAP Feature Attribution)
8. Automated SOAR & Mitigation Layer (iptables/nftables packet drops, quarantine, Merkle Ledger commitment)
"""
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import threading
from itertools import groupby
from typing import List, Optional, Dict, Any
import numpy as np
import torch

from capture.sniffer import PacketRingBuffer, PacketSniffer, RawPacket
from capture.filter import NoiseFilter
from pruning.graph_pruner import AdaptiveGraphPruner
from features.flow_aggregator import FlowAggregator, FlowRecord
from graph.graph_formatter import GraphFormatter, FormattedGraphBatch
from engine.stateful_tgn import StatefulCyberTGN, ThreatAlert
from forecasting.window_forecaster import WindowForecaster
from training.data import sha256
from calibration.conformal import MultiClassSplitConformal
from calibration.artifact import load_calibration
from soar.mitigation import FirewallMitigator, MitigationAction
from soar.policy import SOARPolicyEngine
from forensics.pcap_dumper import ForensicPCAPDumper
from forensics.hasher import compute_alert_state_hash
from forensics.merkle_tree import MerkleTree
from ledger.blockchain_ledger import BlockchainLedger, LedgerBlock
from casestudies.threat_scenarios import ThreatScenario

logger = logging.getLogger("Pipeline.Orchestrator")


@dataclass
class IncidentReport:
    alert: ThreatAlert
    pcap_path: Optional[Path]
    pcap_hash: str
    state_hash: str
    mitigations_taken: List[MitigationAction]
    shap_explanation: Optional[Dict[str, Any]] = None
    subgraph_explanation: Optional[Dict[str, Any]] = None
    conformal_details: Optional[Dict[str, Any]] = None
    future_forecast: Optional[Dict[str, Any]] = None


@dataclass
class PipelineCycleResult:
    flows_processed: int
    alerts_generated: int
    incidents: List[IncidentReport]
    ledger_block: Optional[LedgerBlock] = None
    forecast_results: Optional[Dict[str, Any]] = None


class CyberDefensePipeline:
    """Master orchestrator executing the 8-phase autonomous detection and response architecture."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        dry_run_firewall: bool = True,
        conformal_alpha: float = 0.05,
        alert_threshold: float = 0.85,
        flow_timeout_sec: float = 5.0,
        forensics_dir: Path = Path("forensics/dumps"),
        ledger_file: Path = Path("ledger/ledger.jsonl"),
        forecast_checkpoint: Optional[Path] = None,
        explanations_per_batch: int = 3,
        explanation_epochs: int = 30,
        window_observer=None,
        calibration_file: Optional[Path] = None,
        enable_incidents: bool = True,
    ):
        # 1. Layer 1: Capture & Adaptive Pruning
        self.ring_buffer = PacketRingBuffer(capacity=50000)
        self.sniffer = PacketSniffer(ring_buffer=self.ring_buffer)
        self.noise_filter = NoiseFilter(filter_dns=False)
        self.graph_pruner = AdaptiveGraphPruner(drop_internal_dns=False)
        if explanations_per_batch < 0 or not 1 <= explanation_epochs <= 500:
            raise ValueError('Invalid explanation budget')
        self.explanations_per_batch = explanations_per_batch
        self.explanation_epochs = explanation_epochs
        self.lock = threading.RLock()
        self._window_index = None
        self._window_nodes = set()
        self.window_observer = window_observer
        self.enable_incidents = enable_incidents
        self._pending_flows = []
        self._last_packet_timestamp = None

        # 2. Layer 2: Feature Extraction & Graph Formatting
        self.flow_aggregator = FlowAggregator(flow_timeout_sec=flow_timeout_sec)
        self.graph_formatter = GraphFormatter()

        # 3. AI/ML Engine: Stateful TGN
        self.tgn_engine = StatefulCyberTGN(
            checkpoint_path=checkpoint_path or Path(__file__).resolve().parent / "checkpoints/tgn_best.pt",
            conformal_alpha=conformal_alpha,
            alert_threshold=alert_threshold,
            capture_explanation_context=explanations_per_batch > 0,
        )

        # 4. Phase 4: Delta Forecasting & Kalman Re-Anchoring
        self.graph_formatter.max_nodes = self.tgn_engine.num_nodes
        self.encoder_sha256 = sha256(self.tgn_engine.checkpoint_path)
        state_dim = self.tgn_engine.memory.memory.memory.shape[1]
        self.forecaster = WindowForecaster(state_dim, self.encoder_sha256, forecast_checkpoint)
        self._latest_forecast = dict(status='awaiting_closed_window', forecasts=[], deployment_approved=False)

        # 5. Phase 6: Multi-Class Split Conformal Predictor & ECE
        self.conformal_calibrator = MultiClassSplitConformal(alpha=conformal_alpha)
        self.calibration_protocol = dict(name='stateful_completed_flows_v2', max_batch=1,
                                        history_capacity=self.tgn_engine.history_capacity,
                                        timestamp_encoding=self.tgn_engine.timestamp_encoding,
                                        score='max_graph_entropy_syn', window_seconds=15)
        self.calibration_metadata = None
        if calibration_file is not None:
            binary, stage, metadata = load_calibration(
                calibration_file, self.encoder_sha256, self.calibration_protocol,
                self.tgn_engine.num_classes, conformal_alpha)
            self.tgn_engine.conformal = binary
            self.conformal_calibrator = stage
            self.calibration_metadata = metadata

        # 7. Layer 3: SOAR Mitigation
        self.mitigator = FirewallMitigator(dry_run=dry_run_firewall)
        self.soar_policy = SOARPolicyEngine(mitigator=self.mitigator)

        # 8. Layer 4: Forensics & Immutable Ledger
        self.pcap_dumper = ForensicPCAPDumper(output_dir=forensics_dir, ring_buffer=self.ring_buffer)
        self.ledger = BlockchainLedger(ledger_file=ledger_file)

        logger.info("Experimental CyberDefensePipeline initialized; readiness requires independent validation.")

    def ingest_packet(self, pkt: RawPacket) -> Optional[PipelineCycleResult]:
        with self.lock:
            return self._ingest_packet(pkt)

    def _ingest_packet(self, pkt: RawPacket) -> Optional[PipelineCycleResult]:
        """Ingest a single packet from live sniffing or PCAP. Returns cycle result if flow completed."""
        if (not np.isfinite(pkt.timestamp) or pkt.timestamp < 0 or
            (self._last_packet_timestamp is not None and pkt.timestamp < self._last_packet_timestamp)):
            raise ValueError('Packet event times must be finite, nonnegative and chronological')
        self._last_packet_timestamp = pkt.timestamp
        # Capture callbacks and replay already append to the shared buffer.
        with self.ring_buffer.lock:
            if not self.ring_buffer.buffer or self.ring_buffer.buffer[-1] is not pkt:
                self.ring_buffer.buffer.append(pkt)
        # Expire before appending: a new packet must not revive an idle flow.
        completed = self.flow_aggregator.flush_expired(current_time=pkt.timestamp)
        if not self.noise_filter.is_noise(pkt) and self.graph_pruner.filter_packet(pkt) is not None:
            flow = self.flow_aggregator.add_packet(pkt)
            if flow:
                completed.append(flow)
        self._pending_flows.extend(completed)
        return self._drain_pending(pkt.timestamp)

    def tick(self, timestamp):
        """Flush idle flows and advance event-time windows without a new packet."""
        with self.lock:
            if (not np.isfinite(timestamp) or timestamp < 0 or
                (self._last_packet_timestamp is not None and timestamp < self._last_packet_timestamp)):
                raise ValueError('Tick must not precede the latest packet')
            self._pending_flows.extend(self.flow_aggregator.flush_expired(timestamp))
            result = self._drain_pending(timestamp)
            return result or PipelineCycleResult(0, 0, [], forecast_results=self._latest_forecast)

    def _drain_pending(self, timestamp):
        if len(self._pending_flows) > 10000:
            raise BufferError('Completed-flow reorder buffer exceeded 10000 flows')
        # Idle flows can finish later than newer FIN/RST flows. Hold completed
        # flows until the idle-time watermark makes their ordering final.
        watermark = timestamp-self.flow_aggregator.flow_timeout_sec
        ready = [flow for flow in self._pending_flows if flow.last_time <= watermark]
        self._pending_flows = [flow for flow in self._pending_flows if flow.last_time > watermark]
        result = self.process_flows(ready) if ready else None
        self.advance_watermark(watermark)
        if result is not None:
            result.forecast_results = self._latest_forecast
        return result

    def advance_watermark(self, timestamp):
        """Close an observed window using an actual event/clock watermark."""
        with self.lock:
            if not np.isfinite(timestamp):
                raise ValueError('Watermark must be finite')
            if self._window_index is None or timestamp < (self._window_index+1)*15:
                return self._latest_forecast
            boundary = (self._window_index+1)*15
            if any(flow.last_time < boundary for flow in self._pending_flows):
                return self._latest_forecast
            if any(flow.last_time < boundary for flow in self.flow_aggregator.active_flows.values()):
                return self._latest_forecast
            with self.tgn_engine.lock, torch.no_grad():
                memory, _ = self.tgn_engine.memory(torch.tensor(sorted(self._window_nodes), dtype=torch.long))
                state = memory.mean(0).numpy().copy()
            index = self._window_index
            result = self.forecaster.observe_window(index, state, timestamp)
            if self.window_observer is not None:
                self.window_observer(index, state.copy(), float(timestamp))
            self._latest_forecast = result
            self._window_index = None
            self._window_nodes.clear()
            return result

    def process_flows(self, flows: List[FlowRecord]) -> PipelineCycleResult:
        with self.lock:
            if not flows:
                return PipelineCycleResult(0, 0, [], forecast_results=self._latest_forecast)
            if any(not np.isfinite(f.last_time) or f.last_time < 0 for f in flows):
                raise ValueError('Flow times must be finite and nonnegative')
            ordered = sorted(flows, key=lambda flow: flow.last_time)
            earliest = int(ordered[0].last_time//15)
            last_closed = self.forecaster.previous_window
            if ((last_closed is not None and earliest <= last_closed) or
                (self._window_index is not None and earliest < self._window_index)):
                raise ValueError('Late completed flow belongs to an already processed window')
            cycles = []
            self._explanations_remaining = self.explanations_per_batch
            for index, group in groupby(ordered, key=lambda flow: int(flow.last_time//15)):
                group = list(group)
                self.advance_watermark(group[0].last_time)
                self._window_index = index
                # Fixed batching keeps window memory independent of caller chunking.
                # This serving protocol requires its own accuracy calibration.
                limit = 1
                for start in range(0, len(group), limit):
                    cycles.append(self._process_flows(group[start:start+limit]))
            return PipelineCycleResult(sum(c.flows_processed for c in cycles),
                                       sum(c.alerts_generated for c in cycles),
                                       [i for c in cycles for i in c.incidents],
                                       next((c.ledger_block for c in reversed(cycles) if c.ledger_block), None),
                                       self._latest_forecast)

    def _process_flows(self, flows: List[FlowRecord]) -> PipelineCycleResult:
        """Execute feature engineering, ML scoring, forecasting, SOAR, and forensics."""
        if not flows:
            return PipelineCycleResult(flows_processed=0, alerts_generated=0, incidents=[])

        # Layer 2 & Phase 2: Format dynamic temporal graph batch
        batch = self.graph_formatter.format_flows(flows)

        # Phase 3: Stateful CyberTGN forward pass & base conformal check
        records, alerts = self.tgn_engine.process_graph_batch(batch)
        if not self.enable_incidents:
            alerts = []

        self._window_nodes.update(batch.src_nodes.tolist())
        self._window_nodes.update(batch.dst_nodes.tolist())
        forecast_results = self._latest_forecast
        snapshot = self.tgn_engine.last_explanation_context
        flow_indices = {flow.flow_id: index for index, flow in enumerate(flows)}

        incidents: List[IncidentReport] = []
        state_hashes: List[str] = []

        for alert in alerts:
            shap_info = dict(status='explanation_budget_exhausted', method='permutation_shapley')
            subgraph_info = dict(status='explanation_budget_exhausted', method='pyg_gnnexplainer', highlighted_edges=[])
            if snapshot is not None and self._explanations_remaining > 0:
                self._explanations_remaining -= 1
                index = flow_indices[alert.flow_id]
                try:
                    shap_info = snapshot.explain_features(index)
                    subgraph_info = snapshot.explain_edges(index, epochs=self.explanation_epochs)
                except (RuntimeError, ValueError) as exc:
                    logger.warning('Model explanation unavailable: %s', exc)
                    shap_info = dict(status='failed', method='permutation_shapley', reason=str(exc))
                    subgraph_info = dict(status='failed', method='pyg_gnnexplainer', reason=str(exc), highlighted_edges=[])
                subgraph_info['target_node'] = alert.dst_ip
                for edge in subgraph_info['highlighted_edges']:
                    edge['src_ip'] = self.graph_formatter.id_to_ip[edge['src_node']]
                    edge['dst_ip'] = self.graph_formatter.id_to_ip[edge['dst_node']]

            # Phase 6: Multi-Class Conformal Prediction Set
            # Preserve all checkpoint classes and the actual classifier distribution.
            stage_probs = np.asarray(alert.extra["stage_probabilities"], dtype=np.float64)
            conformal_set_info = self.conformal_calibrator.predict_conformal_set(stage_probs)

            alert.extra['stage_conformal'] = conformal_set_info

            # Layer 3 / Phase 8: Automated SOAR Mitigation (iptables/eBPF drop)
            mitigations = self.soar_policy.handle_alert(alert)

            # Layer 4: Forensic PCAP Dump
            pcap_path, pcap_hash = self.pcap_dumper.dump_alert_pcap(alert)

            # Layer 4 / Phase 8: SHA-256 State Hash
            state_hash = compute_alert_state_hash(alert, pcap_hash)
            state_hashes.append(state_hash)

            incident = IncidentReport(
                alert=alert,
                pcap_path=pcap_path,
                pcap_hash=pcap_hash,
                state_hash=state_hash,
                mitigations_taken=mitigations,
                shap_explanation=shap_info,
                subgraph_explanation=subgraph_info,
                conformal_details=conformal_set_info,
                future_forecast=forecast_results,
            )
            incidents.append(incident)

        # Layer 4 / Phase 8: Merkle Tree Commitment & Blockchain Ledger Block
        ledger_block = None
        if state_hashes:
            ledger_block = self.ledger.commit_batch(state_hashes)
            logger.info(
                f"[LEDGER COMMIT] Block #{ledger_block.block_index} committed | "
                f"Merkle Root: {ledger_block.merkle_root[:16]}... | Leaves: {ledger_block.leaf_count}"
            )

        return PipelineCycleResult(
            flows_processed=len(flows),
            alerts_generated=len(alerts),
            incidents=incidents,
            ledger_block=ledger_block,
            forecast_results=forecast_results,
        )

    def flush_and_process(self) -> PipelineCycleResult:
        """Flush all remaining active flows in memory and process them."""
        with self.lock:
            flushed = self._pending_flows + self.flow_aggregator.flush_all()
            self._pending_flows = []
            return self.process_flows(flushed)

    def replay_pcap(self, pcap_path: str) -> List[PipelineCycleResult]:
        """Replay a PCAP file end-to-end through the pipeline."""
        results = []
        for pkt in self.sniffer.read_pcap(pcap_path):
            res = self.ingest_packet(pkt)
            if res and res.flows_processed > 0:
                results.append(res)

        final_res = self.flush_and_process()
        if final_res and final_res.flows_processed > 0:
            results.append(final_res)

        return results

    def execute_threat_scenario(self, scenario: ThreatScenario) -> PipelineCycleResult:
        """Executes a real-world threat case study scenario end-to-end."""
        logger.info(f"[SCENARIO START] Running case study: {scenario.name} (Target: {scenario.target_cve})")
        cycles = []
        for pkt in scenario.packets:
            cycle = self.ingest_packet(pkt)
            if cycle:
                cycles.append(cycle)
        cycles.append(self.flush_and_process())
        result = PipelineCycleResult(
            flows_processed=sum(c.flows_processed for c in cycles),
            alerts_generated=sum(c.alerts_generated for c in cycles),
            incidents=[incident for c in cycles for incident in c.incidents],
            ledger_block=next((c.ledger_block for c in reversed(cycles) if c.ledger_block), None),
            forecast_results=next((c.forecast_results for c in reversed(cycles) if c.forecast_results), None),
        )
        logger.info(
            f"[SCENARIO COMPLETE] {scenario.name} finished | "
            f"Flows: {result.flows_processed} | Alerts: {result.alerts_generated} | "
            f"Incidents: {len(result.incidents)}"
        )
        return result
