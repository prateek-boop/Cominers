"""Live network packet sniffer connecting raw packet captures to the flow and detection pipeline."""
from __future__ import annotations
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any, List

from engine.flow_engine import FlowEngine, RawPacketInfo
from engine.hybrid_detector import HybridDetector, UnifiedThreatAlert
from engine.mitigation import MitigationController

logger = logging.getLogger(__name__)


class LiveTrafficSniffer:
    """Background packet sniffer with real-time flow aggregation and hybrid scoring."""

    def __init__(self, detector: HybridDetector, mitigation: MitigationController,
                 interface: Optional[str] = None, bpf_filter: str = "ip"):
        self.detector = detector
        self.mitigation = mitigation
        self.interface = interface
        self.bpf_filter = bpf_filter

        self.flow_engine = FlowEngine(idle_timeout_sec=5.0, active_timeout_sec=30.0)
        self.is_running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._async_sniffer = None

        # Callbacks for streaming events to WebSockets or logs
        self.on_flow_scored: Optional[Callable[[UnifiedThreatAlert], None]] = None

        # Telemetry
        self.packets_captured = 0
        self.flows_scored = 0
        self.threats_detected = 0
        self.start_time: Optional[float] = None
        self.last_error = None

    def start(self):
        """Starts live packet capture in a background thread."""
        if self.is_running or (self._thread and self._thread.is_alive()):
            return

        self.is_running = True
        self._stop_event.clear()
        self.start_time = time.time()
        self.last_error = None

        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CyberTGN-Sniffer")
        self._thread.start()
        logger.info(f"Live traffic sniffer started on interface: {self.interface or 'default'}")

    def stop(self):
        """Stops live capture."""
        if not self.is_running and not (self._thread and self._thread.is_alive()):
            return

        self.is_running = False
        self._stop_event.set()

        if self._async_sniffer:
            try:
                self._async_sniffer.stop()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        logger.info("Live traffic sniffer stopped.")

    def _packet_callback(self, scapy_pkt):
        self.packets_captured += 1
        raw_pkt = FlowEngine.parse_scapy_packet(scapy_pkt)
        if raw_pkt:
            completed_flows = self.flow_engine.process_packet(raw_pkt)
            if completed_flows:
                self._score_and_dispatch(completed_flows)

    def _score_and_dispatch(self, flows: List[Dict[str, Any]]):
        if not flows:
            return

        alerts = self.detector.analyze_flows(flows)
        for alert in alerts:
            self.flows_scored += 1
            if alert.is_attack:
                self.threats_detected += 1
                # Trigger mitigation
                self.mitigation.evaluate_and_respond(alert)

            # Broadcast to real-time subscribers
            if self.on_flow_scored:
                try:
                    self.on_flow_scored(alert)
                except Exception as e:
                    logger.debug(f"Error in on_flow_scored callback: {e}")

    def _capture_loop(self):
        try:
            from scapy.all import AsyncSniffer
            self._async_sniffer = AsyncSniffer(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._packet_callback,
                store=False
            )
            self._async_sniffer.start()

            while not self._stop_event.is_set():
                if self._stop_event.wait(1.0):
                    break
                if not self._async_sniffer.running:
                    self._async_sniffer.join()
                    raise RuntimeError("Packet capture stopped unexpectedly")
                # Periodically flush idle flows
                flushed = self.flow_engine.flush_expired(time.time())
                if flushed:
                    self._score_and_dispatch(flushed)

        except Exception as e:
            logger.error(f"Live sniffer encounter error: {e}", exc_info=True)
            self.last_error = str(e)
        finally:
            if self._async_sniffer and self._async_sniffer.running:
                try:
                    self._async_sniffer.stop()
                except Exception:
                    logger.exception("Failed to stop capture")
            try:
                self._score_and_dispatch(self.flow_engine.flush_all())
            finally:
                self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time if self.start_time and self.is_running else 0.0
        return {
            "is_running": self.is_running,
            "last_error": self.last_error,
            "interface": self.interface or "default",
            "uptime_seconds": round(uptime, 1),
            "packets_captured": self.packets_captured,
            "flows_scored": self.flows_scored,
            "threats_detected": self.threats_detected,
            "active_flows_in_memory": len(self.flow_engine.active_flows),
        }
