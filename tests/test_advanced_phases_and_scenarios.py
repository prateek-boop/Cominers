"""Tests for Advanced 8-Phase Architecture and Real-World Threat Case Studies.

Validates:
- Phase 1: Adaptive Graph Pruner (NTP/DNS/ARP background drop)
- Phase 4: Delta Formulation & Discrete Kalman Filter Re-Anchoring
- Phase 6: Split Conformal Multi-Class Prediction & Expected Calibration Error (ECE)
- Real-World Threat Scenarios:
  * SolarWinds SUNBURST (Build pipeline injection, valid cert, 2-week dwell, DNS C2)
  * Salt Typhoon (Telecom core router compromise, Lawful Intercept wiretap lateral movement)
  * Volt Typhoon (Cisco RV320/RV325 CVE-2019-1653 config dump & synthetic command-injection step (no CVE attribution) command injection)
"""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from capture.sniffer import RawPacket
from pruning.graph_pruner import AdaptiveGraphPruner
from forecasting.delta_predictor import DeltaPredictor
from forecasting.kalman_filter import StateKalmanFilter
from calibration.conformal import MultiClassSplitConformal
from calibration.ece import compute_ece
from casestudies.threat_scenarios import (
    generate_solarwinds_sunburst_scenario,
    generate_salt_typhoon_telecom_scenario,
    generate_volt_typhoon_rv320_scenario,
)
from pipeline import CyberDefensePipeline


class AdvancedArchitectureTests(unittest.TestCase):

    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)

    def make_pipeline(self):
        root = Path(self.output.name)
        return CyberDefensePipeline(dry_run_firewall=True, forensics_dir=root / "pcaps", ledger_file=root / "ledger.jsonl")

    def test_phase1_adaptive_graph_pruner(self):
        """Verify that routine benign background protocols (NTP, ARP, internal DNS) are dropped."""
        pruner = AdaptiveGraphPruner()

        p_ntp = RawPacket(1700000000.0, "10.0.0.1", "pool.ntp.org", 123, 123, "udp", 48)
        p_arp = RawPacket(1700000001.0, "10.0.0.1", "255.255.255.255", 0, 0, "arp", 42)
        p_dns = RawPacket(1700000002.0, "10.0.0.5", "10.0.0.53", 53000, 53, "udp", 80)
        p_tcp = RawPacket(1700000003.0, "10.0.0.5", "10.0.0.10", 40000, 445, "tcp", 120, tcp_flags=0x02)

        self.assertIsNone(pruner.filter_packet(p_ntp))
        self.assertIsNone(pruner.filter_packet(p_arp))
        self.assertIsNone(pruner.filter_packet(p_dns))
        self.assertIsNotNone(pruner.filter_packet(p_tcp))

    def test_phase4_delta_forecasting_and_kalman(self):
        """Verify residual delta predictor and discrete Kalman re-anchoring."""
        predictor = DeltaPredictor(state_dim=128, hidden_dim=64, num_layers=1)
        current_state = torch.randn(1, 128)

        # 1. Delta Prediction
        out = predictor(current_state)
        self.assertIn("delta_15s", out)
        self.assertIn("forecast_15s", out)
        self.assertIn("forecast_30s", out)
        self.assertIn("forecast_45s", out)

        # Mathematical property: forecast = current + delta
        diff = torch.norm((current_state + out["delta_15s"]) - out["forecast_15s"])
        self.assertAlmostEqual(float(diff), 0.0, places=5)

        # 2. Discrete Kalman Re-Anchoring
        kalman = StateKalmanFilter(state_dim=128)
        pred_flat = out["forecast_15s"].squeeze(0).detach().numpy()
        s_obs = current_state.squeeze(0).numpy() + np.random.randn(128) * 0.1

        # Re-anchor with observation
        corrected = kalman.reanchor_and_correct(pred_flat, s_obs)
        self.assertEqual(corrected.shape, (128,))

        # Subsequent rollout from anchored state
        rolled = kalman.predict_next_anchored(out["delta_30s"].squeeze(0).detach().numpy())
        self.assertEqual(rolled.shape, (128,))

    def test_phase6_calibration_conformal_and_ece(self):
        """Verify split conformal multi-class prediction set and Expected Calibration Error."""
        # 1. Conformal Predictor
        conformal = MultiClassSplitConformal(alpha=0.10)
        # Synthetic calibration set
        y_cal = np.array([0, 1, 4, 5, 0, 2])
        probs_cal = np.random.dirichlet(np.ones(8), size=6)
        conformal.calibrate(y_cal, probs_cal)
        self.assertTrue(conformal.is_calibrated)
        self.assertTrue(0.0 <= conformal.q_hat <= 1.0)

        # Predict set for an ambiguous test instance
        test_probs = np.array([0.05, 0.10, 0.05, 0.05, 0.40, 0.30, 0.03, 0.02])
        res = conformal.predict_conformal_set(test_probs)
        self.assertIn("conformal_set", res)
        self.assertGreaterEqual(res["set_size"], 1)

        # 2. Expected Calibration Error (ECE)
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.65, 0.3, 0.15, 0.85, 0.25])
        ece_res = compute_ece(y_true, y_prob, n_bins=5)
        self.assertTrue(0.0 <= ece_res["ece"] <= 1.0)
        self.assertEqual(len(ece_res["bins"]), 5)

    def test_case_study_solarwinds_sunburst(self):
        """Replay SolarWinds SUNBURST scenario through full 8-phase pipeline."""
        pipeline = self.make_pipeline()
        scenario = generate_solarwinds_sunburst_scenario()
        res = pipeline.execute_threat_scenario(scenario)

        self.assertGreater(res.flows_processed, 0)
        self.assertIn("SolarWinds", scenario.name)
        self.assertEqual(scenario.target_cve, "CVE-2020-10148")

    def test_case_study_salt_typhoon_telecom(self):
        """Replay Salt Typhoon Telecom Core Lawful Intercept scenario through full pipeline."""
        pipeline = self.make_pipeline()
        scenario = generate_salt_typhoon_telecom_scenario()
        res = pipeline.execute_threat_scenario(scenario)

        self.assertGreater(res.flows_processed, 0)
        self.assertIn("SaltTyphoon", scenario.name)

    def test_case_study_volt_typhoon_cisco_rv320_rv325(self):
        """Replay Volt Typhoon Cisco RV320/RV325 (CVE-2019-1653-inspired synthetic traffic) scenario through full pipeline."""
        pipeline = self.make_pipeline()
        scenario = generate_volt_typhoon_rv320_scenario()
        res = pipeline.execute_threat_scenario(scenario)

        self.assertGreater(res.flows_processed, 0)
        self.assertIn("VoltTyphoon", scenario.name)
        self.assertIn("CVE-2019-1653", scenario.target_cve)


if __name__ == '__main__':
    unittest.main()
