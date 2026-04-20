#!/usr/bin/env python3
"""
CTF Integration Layer — LeKiWi ↔ Robot-Security-Workshop
=========================================================
Bridges the lekiwi_vla CTF monitoring system with the robot-security-workshop
CTF scoring platform.

Architecture:
  LeKiWi Bridge (ctf_mode=True)
      ↓ /lekiwi/security_alert
  CTFSecurityAuditor → SecurityAlert
      ↓ alert_callback
  CTFIntegrationHub
      ↓ REST API
  robot-security-workshop CTF Platform (Flask)
      ↓
  Participant submits flag → CTF scoreboard

Usage:
  # Start bridge with CTF integration:
  ros2 launch lekiwi_ros2_bridge bridge.launch.py ctf_mode:=true

  # Start CTF integration hub (separate process):
  python3 ctf_integration.py --mode hub --ctf-url http://localhost:5000

  # Simulate attack for testing:
  python3 ctf_integration.py --mode attacker --challenge C1 --target 192.168.1.100

CTF Challenges Mapped to ROS2 Topics:
  C1 (Teleport):       /lekiwi/cmd_vel         → forged twist
  C2 (Eavesdrop):       /lekiwi/joint_states     → plaintext leak
  C3 (Bypass Auth):     /lekiwi/service/*        → unprotected service
  C4 (Serial Shell):    /dev/ttyUSB0             → UART debug
  C5 (Firmware Dump):   JTAG interface           → firmware extraction
  C6 (Adversarial):     /lekiwi/camera/image_raw → adversarial patches
  C7 (Policy Hijack):   /lekiwi/policy           → ByteMultiArray injection
  C8 (Policy Inject):   /lekiwi/vla_action       → action override

Security Alert → CTF Flag Mapping:
  C1_hmac_missing  → ROBOT_CTF{cmdvel_hmac_missing_a1b2c3d4}
  C2_rate_flood    → ROBOT_CTF{cmdvel_dos_rate_flood_e5f6g7h8}
  C3_injection     → ROBOT_CTF{cmdvel_injection_i9j0k1l2}
  C4_physics_dos   → ROBOT_CTF{physics_dos_accel_m3n4o5p6}
  C5_replay        → ROBOT_CTF{replay_attack_q7r8s9t0}
  C6_sensor_spoof  → ROBOT_CTF{sensor_spoof_u1v2w3x4}
  C7_policy_inject → ROBOT_CTF{policy_inject_y5z6a7b8}
  C8_policy_hijack → ROBOT_CTF{policy_hijack_c9d0e1f2}

Integration with robot-security-workshop:
  - CTFPlatformClient: REST client for submitting flags to Flask scoreboard
  - CTFAlertForwarder: Forwards SecurityAlert → CTF platform webhook
  - CTFAttackSimulator: Simulates attacks for platform testing
  - CTFEventLogger: Logs security events to JSONL for forensic analysis
"""

import argparse
import json
import time
import socket
import struct
import threading
import subprocess
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict, field
from collections import deque
import requests

# ── CTF Platform Client ──────────────────────────────────────────────────────

CTF_FLAGS = {
    "C1": "ROBOT_CTF{cmdvel_hmac_missing_a1b2c3d4}",
    "C2": "ROBOT_CTF{cmdvel_dos_rate_flood_e5f6g7h8}",
    "C3": "ROBOT_CTF{cmdvel_injection_i9j0k1l2}",
    "C4": "ROBOT_CTF{physics_dos_accel_m3n4o5p6}",
    "C5": "ROBOT_CTF{replay_attack_q7r8s9t0}",
    "C6": "ROBOT_CTF{sensor_spoof_u1v2w3x4}",
    "C7": "ROBOT_CTF{policy_inject_y5z6a7b8}",
    "C8": "ROBOT_CTF{policy_hijack_c9d0e1f2}",
}

CHALLENGE_NAMES = {
    "C1": "challenge_01_teleport",
    "C2": "challenge_02_eavesdrop",
    "C3": "challenge_03_bypass_auth",
    "C4": "challenge_04_serial_shell",
    "C5": "challenge_05_firmware_dump",
    "C6": "challenge_06_adversarial",
    "C7": "challenge_07_policy_hijack",
    "C8": "challenge_08_policy_inject",
}

CHALLENGE_CATEGORIES = {
    "C1": "Network",
    "C2": "Network",
    "C3": "Network",
    "C4": "Hardware",
    "C5": "Hardware",
    "C6": "AI Security",
    "C7": "AI Security",
    "C8": "AI Security",
}


@dataclass
class CTFEvent:
    """A security event mapped to a CTF challenge."""
    challenge_id: str           # "C1"-"C8"
    timestamp: float
    severity: str               # "low", "medium", "high", "critical"
    description: str
    source_ip: Optional[str] = None
    raw_data: Optional[dict] = None
    flag_captured: bool = False
    flag: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["challenge_name"] = CHALLENGE_NAMES.get(self.challenge_id, "unknown")
        d["category"] = CHALLENGE_CATEGORIES.get(self.challenge_id, "Unknown")
        d["flag"] = self.flag or CTF_FLAGS.get(self.challenge_id)
        return d


class CTFPlatformClient:
    """
    REST client for the robot-security-workshop CTF Flask platform.
    Submits flags and retrieves challenge metadata.
    """

    def __init__(self, base_url: str = "http://localhost:5000", team_name: str = "lekiwi_system"):
        self.base_url = base_url.rstrip("/")
        self.team_name = team_name
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def submit_flag(self, challenge_id: str, flag: str) -> dict:
        """
        Submit a flag to the CTF scoring platform.
        
        Returns: {"success": bool, "points": int, "message": str}
        """
        endpoint = f"{self.base_url}/api/flag/submit"
        payload = {
            "team_name": self.team_name,
            "challenge_id": challenge_id,
            "flag": flag,
        }
        try:
            resp = self.session.post(endpoint, json=payload, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_challenges(self) -> dict:
        """Retrieve all available challenges from the platform."""
        try:
            resp = self.session.get(f"{self.base_url}/api/challenges", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_scoreboard(self) -> dict:
        """Retrieve current scoreboard."""
        try:
            resp = self.session.get(f"{self.base_url}/api/scoreboard", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def test_connection(self) -> bool:
        """Check if CTF platform is reachable."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ── CTF Integration Hub ───────────────────────────────────────────────────────

class CTFIntegrationHub:
    """
    Central hub that connects LeKiWi bridge security alerts to the CTF platform.

    Receives SecurityAlert from bridge_node._on_security_alert callback,
    maps them to CTF challenges, and forwards to the CTF scoring platform.

    Also handles:
    - Event logging (JSONL)
    - Attack simulation for testing
    - Webhook forwarding to external SIEM
    """

    def __init__(
        self,
        ctf_url: str = "http://localhost:5000",
        team_name: str = "lekiwi_system",
        log_path: Optional[str] = None,
        forward_webhook: Optional[str] = None,
    ):
        self.ctf_client = CTFPlatformClient(base_url=ctf_url, team_name=team_name)
        self.forward_webhook = forward_webhook
        self.team_name = team_name

        # Event storage
        self.events: deque[CTFEvent] = deque(maxlen=1000)
        self.captured_flags: set[str] = set()
        self.alert_callback: Optional[Callable[[CTFEvent], None]] = None

        # Logging
        if log_path:
            self._log_path = Path(log_path)
        else:
            self._log_path = Path.home() / "hermes_research" / "lekiwi_vla" / "ctf_events.jsonl"
        self._log_file = open(self._log_path, "a", buffering=1)

        # Statistics
        self.stats = {
            "total_events": 0,
            "flags_captured": set(),
            "by_challenge": {cid: 0 for cid in CTF_FLAGS},
            "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
        }

    def on_security_alert(self, alert_data: dict) -> None:
        """
        Process a SecurityAlert from bridge_node._on_security_alert.
        alert_data is the JSON-serialized SecurityAlert payload.
        """
        try:
            challenge_id = self._map_alert_to_challenge(alert_data)
            if challenge_id is None:
                return  # Not a CTF-relevant alert

            event = CTFEvent(
                challenge_id=challenge_id,
                timestamp=alert_data.get("timestamp", time.time()),
                severity=alert_data.get("severity", "high"),
                description=alert_data.get("description", ""),
                source_ip=alert_data.get("source_ip"),
                raw_data=alert_data,
                flag_captured=False,
                flag=CTF_FLAGS.get(challenge_id),
            )

            self._record_event(event)

            # Forward to CTF platform if flag was captured
            if event.flag_captured and event.flag:
                self._submit_to_ctf(event)

            # Forward to webhook if configured
            if self.forward_webhook:
                self._forward_webhook(event)

        except Exception as e:
            print(f"[CTFHub] Error processing alert: {e}")

    def simulate_attack(self, challenge_id: str, target_host: str = "127.0.0.1") -> CTFEvent:
        """
        Simulate an attack for a given challenge.
        Returns the generated CTFEvent.
        """
        print(f"[CTFHub] Simulating {challenge_id} attack against {target_host}...")

        if challenge_id == "C1":
            # Challenge 1: Forge cmd_vel without HMAC
            return self._simulate_teleport_attack(target_host)
        elif challenge_id == "C2":
            # Challenge 2: Eavesdrop on ROS2 topic (simulate flag discovery)
            return self._simulate_eavesdrop(target_host)
        elif challenge_id == "C3":
            # Challenge 3: Bypass auth (call unprotected service)
            return self._simulate_auth_bypass(target_host)
        elif challenge_id == "C4":
            # Challenge 4: Serial shell
            return self._simulate_serial_shell(target_host)
        elif challenge_id == "C5":
            # Challenge 5: Firmware dump
            return self._simulate_firmware_dump(target_host)
        elif challenge_id == "C6":
            # Challenge 6: Adversarial patch
            return self._simulate_adversarial(target_host)
        elif challenge_id == "C7":
            # Challenge 7: Policy hijack
            return self._simulate_policy_hijack(target_host)
        elif challenge_id == "C8":
            # Challenge 8: Policy inject
            return self._simulate_policy_inject(target_host)
        else:
            raise ValueError(f"Unknown challenge: {challenge_id}")

    def _simulate_teleport_attack(self, target: str) -> CTFEvent:
        """C1: Send forged cmd_vel with extreme values."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Forge cmd_vel: massive velocity = teleport
            forged = struct.pack('ddd', 100.0, 100.0, 10.0)  # vx, vy, wz
            sock.sendto(forged, (target, 8080))
            sock.close()
            flag = CTF_FLAGS["C1"]
            self.captured_flags.add(flag)
            return CTFEvent(
                challenge_id="C1",
                timestamp=time.time(),
                severity="critical",
                description=f"Forged cmd_vel sent to {target}:8080 — ROBOT_CTF{{teleport_success_6f8d2a1b}}",
                source_ip=target,
                flag_captured=True,
                flag=flag,
            )
        except Exception as e:
            return CTFEvent(
                challenge_id="C1",
                timestamp=time.time(),
                severity="high",
                description=f"Attack failed: {e}",
                source_ip=target,
            )

    def _simulate_eavesdrop(self, target: str) -> CTFEvent:
        """C2: Monitor ROS2 topic for base64 flag."""
        # Simulate: flag found in DDS traffic
        flag = CTF_FLAGS["C2"]
        self.captured_flags.add(flag)
        return CTFEvent(
            challenge_id="C2",
            timestamp=time.time(),
            severity="medium",
            description="Flag ROBOT_CTF{eavesdrop_dds_9c3e7f1a} found in ROS2 DDS traffic",
            source_ip=target,
            flag_captured=True,
            flag=flag,
        )

    def _simulate_auth_bypass(self, target: str) -> CTFEvent:
        """C3: Call unprotected ROS2 service."""
        flag = CTF_FLAGS["C3"]
        self.captured_flags.add(flag)
        return CTFEvent(
            challenge_id="C3",
            timestamp=time.time(),
            severity="high",
            description="Unprotected ROS2 service called — flag: ROBOT_CTF{bypass_service_a2d4c8e1}",
            source_ip=target,
            flag_captured=True,
            flag=flag,
        )

    def _simulate_serial_shell(self, target: str) -> CTFEvent:
        """C4: UART serial shell access."""
        flag = CTF_FLAGS["C4"]
        self.captured_flags.add(flag)
        return CTFEvent(
            challenge_id="C4",
            timestamp=time.time(),
            severity="critical",
            description="UART serial shell accessed (115200 baud, no auth) — flag: ROBOT_CTF{serial_root_5b7e9c3f}",
            source_ip=target,
            flag_captured=True,
            flag=flag,
        )

    def _simulate_firmware_dump(self, target: str) -> CTFEvent:
        """C5: JTAG firmware extraction."""
        flag = CTF_FLAGS["C5"]
        self.captured_flags.add(flag)
        return CTFEvent(
            challenge_id="C5",
            timestamp=time.time(),
            severity="critical",
            description="JTAG firmware dump extracted — flag: ROBOT_CTF{firmware_jtag_8a2d4f6e}",
            source_ip=target,
            flag_captured=True,
            flag=flag,
        )

    def _simulate_adversarial(self, target: str) -> CTFEvent:
        """C6: Adversarial patch on camera feed."""
        flag = CTF_FLAGS["C6"]
        self.captured_flags.add(flag)
        return CTFEvent(
            challenge_id="C6",
            timestamp=time.time(),
            severity="high",
            description="Adversarial patch applied to stop sign detector — flag: ROBOT_CTF{adversarial_ae9f3c7d}",
            source_ip=target,
            flag_captured=True,
            flag=flag,
        )

    def _simulate_policy_hijack(self, target: str) -> CTFEvent:
        """C7: Malicious policy loaded via /lekiwi/policy topic."""
        flag = CTF_FLAGS["C7"]
        self.captured_flags.add(flag)
        return CTFEvent(
            challenge_id="C7",
            timestamp=time.time(),
            severity="critical",
            description="Malicious policy injected via /lekiwi/policy — flag: ROBOT_CTF{policy_hijack_4c8e2a9f}",
            source_ip=target,
            flag_captured=True,
            flag=flag,
        )

    def _simulate_policy_inject(self, target: str) -> CTFEvent:
        """C8: VLA action override via /lekiwi/vla_action."""
        flag = CTF_FLAGS["C8"]
        self.captured_flags.add(flag)
        return CTFEvent(
            challenge_id="C8",
            timestamp=time.time(),
            severity="high",
            description="VLA action override via /lekiwi/vla_action — flag: ROBOT_CTF{policy_inject_y5z6a7b8}",
            source_ip=target,
            flag_captured=True,
            flag=flag,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _map_alert_to_challenge(self, alert: dict) -> Optional[str]:
        """Map SecurityAlert type to CTF challenge ID."""
        alert_type = alert.get("type", "")
        challenge_id = alert.get("challenge_id", "")

        # Direct challenge_id mapping
        if challenge_id in CTF_FLAGS:
            return challenge_id

        # Type-based mapping
        type_to_challenge = {
            "hmac_failed":           "C1",
            "rate_limit":            "C2",
            "magnitude_violation":   "C3",
            "acceleration_violation":"C4",
            "replay_detected":       "C5",
            "sensor_spoof":          "C6",
            "policy_inject":         "C7",
            "policy_hijack":         "C8",
            "goal_spoof":            "C8",   # C8: goal spoofing via /lekiwi/goal
            "goal_rate_limit":       "C8",
            "goal_teleport":         "C8",
            "goal_out_of_bounds":    "C8",
        }
        return type_to_challenge.get(alert_type)

    def on_goal_spoof(self, gx: float, gy: float, reason: str,
                      flag: Optional[str] = None) -> Optional[CTFEvent]:
        """
        Handle detected goal spoofing attack (C8).

        Called by bridge_node._on_goal() when SecurityMonitor detects a
        suspicious /lekiwi/goal update (rate abuse, teleportation, OOB).

        Returns a CTFEvent if this is a new (non-duplicate) attack event.
        """
        event = CTFEvent(
            challenge_id="C8",
            timestamp=time.time(),
            severity="high",
            description=f"[GOAL SPOOF] {reason} — goal=({gx:.3f}, {gy:.3f})",
            raw_data={"gx": gx, "gy": gy, "reason": reason},
            flag_captured=False,
            flag=flag or CTF_FLAGS.get("C8"),
        )
        self._record_event(event)
        return event

    def _record_event(self, event: CTFEvent) -> None:
        """Log event to deque, file, and stats."""
        self.events.append(event)
        self.stats["total_events"] += 1
        self.stats["by_challenge"][event.challenge_id] += 1
        self.stats["by_severity"][event.severity] += 1
        if event.flag_captured:
            self.stats["flags_captured"].add(event.challenge_id)

        # Write JSONL
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        self._log_file.write(line + "\n")

        # Callback
        if self.alert_callback:
            self.alert_callback(event)

        print(f"[CTFHub] {event.challenge_id} [{event.severity}] {event.description}")

    def _submit_to_ctf(self, event: CTFEvent) -> None:
        """Submit captured flag to CTF platform."""
        if not event.flag:
            return
        result = self.ctf_client.submit_flag(event.challenge_id, event.flag)
        if result.get("success"):
            print(f"[CTFHub] ✓ Flag submitted: {event.challenge_id} → +{result.get('points', 0)} pts")
        else:
            print(f"[CTFHub] ✗ Flag submit failed: {result.get('error')}")

    def _forward_webhook(self, event: CTFEvent) -> None:
        """Forward event to external webhook (SIEM, Slack, etc.)."""
        try:
            payload = event.to_dict()
            requests.post(self.forward_webhook, json=payload, timeout=3)
        except requests.RequestException as e:
            print(f"[CTFHub] Webhook forward failed: {e}")

    def get_report(self) -> dict:
        """Generate CTF integration status report."""
        return {
            "ctf_platform": self.ctf_client.base_url,
            "team": self.team_name,
            "log_path": str(self._log_path),
            "stats": {
                "total_events": self.stats["total_events"],
                "flags_captured": list(self.stats["flags_captured"]),
                "by_challenge": self.stats["by_challenge"],
                "by_severity": self.stats["by_severity"],
            },
            "recent_events": [e.to_dict() for e in list(self.events)[-10:]],
        }

    def print_report(self) -> None:
        """Print human-readable CTF integration report."""
        report = self.get_report()
        print("\n" + "=" * 60)
        print("  CTF Integration Hub — Status Report")
        print("=" * 60)
        print(f"  CTF Platform : {report['ctf_platform']}")
        print(f"  Team         : {report['team']}")
        print(f"  Log          : {report['log_path']}")
        print()
        print(f"  Total Events : {report['stats']['total_events']}")
        print(f"  Flags Cap    : {', '.join(report['stats']['flags_captured']) or '(none)'}")
        print()
        print("  By Challenge :")
        for cid, count in report['stats']['by_challenge'].items():
            flag_mark = "✓" if cid in report['stats']['flags_captured'] else " "
            print(f"    [{flag_mark}] {cid}: {count}")
        print()
        print("  By Severity  :")
        for sev, count in report['stats']['by_severity'].items():
            print(f"    {sev:10s}: {count}")
        print()
        print("  Recent Events:")
        for e in report['recent_events']:
            flag_mark = "✓" if e['flag_captured'] else " "
            print(f"    [{flag_mark}] {e['challenge_id']} [{e['severity']:8s}] {e['description'][:50]}")
        print("=" * 60)

    def close(self) -> None:
        """Close log file."""
        if hasattr(self, '_log_file') and self._log_file:
            self._log_file.close()


# ── UDP Attack Simulator ─────────────────────────────────────────────────────

class CTFAttackSimulator:
    """
    Standalone UDP attack simulator for testing CTF challenges.
    Sends various attack payloads to the vulnerable robot.
    """

    def __init__(self, target_host: str = "127.0.0.1", target_port: int = 8080):
        self.target_host = target_host
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def attack_teleport(self) -> bool:
        """C1: Send forged cmd_vel with extreme values."""
        payload = struct.pack('ddd', 100.0, 100.0, 10.0)  # vx, vy, wz (m/s, rad/s)
        self.sock.sendto(payload, (self.target_host, self.target_port))
        print(f"[Attack] C1 Teleport: sent forged cmd_vel {{100, 100, 10}} to {self.target_host}:{self.target_port}")
        return True

    def attack_eavesdrop(self) -> str:
        """
        C2: Simulate flag found in ROS2 DDS traffic.
        Returns the discovered flag.
        """
        flag = "ROBOT_CTF{eavesdrop_dds_9c3e7f1a}"
        print(f"[Attack] C2 Eavesdrop: intercepted flag from ROS2 topic — {flag}")
        return flag

    def attack_bypass_auth(self) -> str:
        """C3: Call unprotected ROS2 service."""
        flag = "ROBOT_CTF{bypass_service_a2d4c8e1}"
        print(f"[Attack] C3 Bypass Auth: called unprotected service — {flag}")
        return flag

    def attack_serial_shell(self) -> str:
        """C4: UART serial shell access."""
        flag = "ROBOT_CTF{serial_root_5b7e9c3f}"
        print(f"[Attack] C4 Serial Shell: connected to /dev/ttyACM0 (115200) — {flag}")
        return flag

    def attack_firmware_dump(self) -> str:
        """C5: JTAG firmware extraction."""
        flag = "ROBOT_CTF{firmware_jtag_8a2d4f6e}"
        print(f"[Attack] C5 Firmware Dump: JTAG extraction complete — {flag}")
        return flag

    def attack_adversarial(self) -> str:
        """C6: Adversarial patch on camera feed."""
        flag = "ROBOT_CTF{adversarial_ae9f3c7d}"
        print(f"[Attack] C6 Adversarial: stop sign patch applied — {flag}")
        return flag

    def attack_policy_hijack(self) -> str:
        """C7: Malicious policy via /lekiwi/policy."""
        flag = "ROBOT_CTF{policy_hijack_4c8e2a9f}"
        print(f"[Attack] C7 Policy Hijack: malicious policy loaded — {flag}")
        return flag

    def attack_policy_inject(self) -> str:
        """C8: VLA action override."""
        flag = "ROBOT_CTF{policy_inject_y5z6a7b8}"
        print(f"[Attack] C8 Policy Inject: action override sent — {flag}")
        return flag

    def run_all(self) -> list[tuple[str, str]]:
        """Run all 8 attacks and return (challenge_id, flag) pairs."""
        results = []
        results.append(("C1", self.attack_teleport()))
        results.append(("C2", self.attack_eavesdrop()))
        results.append(("C3", self.attack_bypass_auth()))
        results.append(("C4", self.attack_serial_shell()))
        results.append(("C5", self.attack_firmware_dump()))
        results.append(("C6", self.attack_adversarial()))
        results.append(("C7", self.attack_policy_hijack()))
        results.append(("C8", self.attack_policy_inject()))
        return results

    def close(self) -> None:
        self.sock.close()


# ── ROS2 Topic Monitor ───────────────────────────────────────────────────────

class CTFROSTopicMonitor:
    """
    Monitor ROS2 topics for CTF-relevant data (flags, sensitive info).
    Uses ros2 CLI or Python ROS2 libraries to subscribe to topics.
    """

    def __init__(self, master_uri: str = "http://localhost:11311"):
        self.master_uri = master_uri
        self.flags_found: list[tuple[str, str]] = []  # (topic, flag)

    def monitor_topic(self, topic: str, duration: float = 5.0) -> list[str]:
        """
        Monitor a single ROS2 topic for base64-encoded flags.
        Returns list of discovered flags.
        """
        print(f"[ROSTopic] Monitoring {topic} for {duration}s...")
        # Use ros2 topic echo in subprocess
        try:
            result = subprocess.run(
                ["ros2", "topic", "echo", topic, "--once"],
                capture_output=True,
                text=True,
                timeout=duration,
                env={**os.environ, "ROS_MASTER_URI": self.master_uri},
            )
            output = result.stdout + result.stderr
            # Look for base64-encoded flags
            import base64
            flags = []
            for line in output.split("\n"):
                if "FLAG:" in line or "flag" in line.lower():
                    # Try to decode base64
                    parts = line.split("FLAG:")[-1].strip()
                    try:
                        decoded = base64.b64decode(parts).decode()
                        if "ROBOT_CTF{" in decoded:
                            flags.append(decoded)
                            self.flags_found.append((topic, decoded))
                    except Exception:
                        pass
            return flags
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[ROSTopic] Monitor failed: {e}")
            return []

    def monitor_all(self, duration: float = 10.0) -> dict[str, list[str]]:
        """Monitor all ROS2 topics for flags."""
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True,
                text=True,
                timeout=5,
                env={**os.environ, "ROS_MASTER_URI": self.master_uri},
            )
            topics = [t.strip() for t in result.stdout.split("\n") if t.strip()]
        except FileNotFoundError:
            print("[ROSTopic] ros2 CLI not found — skipping topic enumeration")
            return {}

        all_flags = {}
        for topic in topics:
            flags = self.monitor_topic(topic, duration=duration)
            if flags:
                all_flags[topic] = flags
        return all_flags


# ── Main CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LeKiWi CTF Integration Layer — bridges lekiwi_vla security monitoring with robot-security-workshop CTF platform"
    )
    parser.add_argument(
        "--mode",
        choices=["hub", "attacker", "monitor", "submit", "report"],
        default="hub",
        help="Mode: hub (run integration hub), attacker (simulate attacks), monitor (ROS2 topic monitor), submit (submit a flag), report (print status)",
    )
    parser.add_argument("--ctf-url", default="http://localhost:5000", help="CTF platform URL")
    parser.add_argument("--team", default="lekiwi_system", help="Team name for scoreboard")
    parser.add_argument("--challenge", help="Challenge ID (C1-C8) for attacker mode")
    parser.add_argument("--flag", help="Flag to submit (for submit mode)")
    parser.add_argument("--target", default="127.0.0.1", help="Target host for attacks")
    parser.add_argument("--webhook", help="Optional webhook URL for SIEM forwarding")
    parser.add_argument("--log-path", help="Path for CTF event log (JSONL)")

    args = parser.parse_args()

    if args.mode == "hub":
        hub = CTFIntegrationHub(
            ctf_url=args.ctf_url,
            team_name=args.team,
            log_path=args.log_path,
            forward_webhook=args.webhook,
        )

        # Test CTF platform connection
        if hub.ctf_client.test_connection():
            print(f"[CTFHub] Connected to CTF platform: {args.ctf_url}")
        else:
            print(f"[CTFHub] WARNING: Cannot reach CTF platform at {args.ctf_url}")

        hub.print_report()

        print("\n[CTFHub] Running — press Ctrl+C to stop")
        print("Waiting for security alerts from bridge_node...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[CTFHub] Shutting down...")
            hub.print_report()
            hub.close()

    elif args.mode == "attacker":
        if not args.challenge:
            print("[Attacker] Running all attack simulations...")
            sim = CTFAttackSimulator(target_host=args.target)
            results = sim.run_all()
            sim.close()
            print("\n[Attacker] Results:")
            for cid, flag in results:
                print(f"  {cid}: {flag}")
        else:
            sim = CTFAttackSimulator(target_host=args.target)
            method = f"attack_{['teleport','eavesdrop','bypass_auth','serial_shell','firmware_dump','adversarial','policy_hijack','policy_inject'][['C1','C2','C3','C4','C5','C6','C7','C8'].index(args.challenge)]}"
            if hasattr(sim, method):
                getattr(sim, method)()
            sim.close()

    elif args.mode == "monitor":
        monitor = CTFROSTopicMonitor()
        print(f"[ROSTopic] Monitoring all topics for 10 seconds...")
        flags = monitor.monitor_all(duration=10.0)
        print(f"\n[ROSTopic] Flags found: {len(flags)}")
        for topic, flag_list in flags.items():
            for f in flag_list:
                print(f"  {topic}: {f}")

    elif args.mode == "submit":
        if not args.flag:
            print("[Submit] Error: --flag required")
            return
        client = CTFPlatformClient(base_url=args.ctf_url, team_name=args.team)
        # Find challenge_id from flag
        challenge_id = None
        for cid, f in CTF_FLAGS.items():
            if f == args.flag:
                challenge_id = cid
                break
        if not challenge_id:
            print(f"[Submit] Unknown flag: {args.flag}")
            return
        result = client.submit_flag(challenge_id, args.flag)
        print(f"[Submit] Result: {result}")

    elif args.mode == "report":
        log_path = args.log_path or str(Path.home() / "hermes_research" / "lekiwi_vla" / "ctf_events.jsonl")
        if not Path(log_path).exists():
            print(f"[Report] No log file at {log_path}")
            return
        events = []
        with open(log_path) as f:
            for line in f:
                events.append(json.loads(line))
        print(f"\n[Report] Total events: {len(events)}")
        flags = [e for e in events if e.get("flag_captured")]
        print(f"[Report] Flags captured: {len(flags)}")
        for e in flags:
            print(f"  {e['challenge_id']}: {e.get('flag', CTF_FLAGS.get(e['challenge_id']))}")


if __name__ == "__main__":
    main()
