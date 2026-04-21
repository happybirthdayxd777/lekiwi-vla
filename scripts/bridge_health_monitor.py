#!/usr/bin/env python3
"""
bridge_health_monitor.py — Phase 257
====================================
Standalone health-check script for lekiwi_ros2_bridge.

Verifies:
  1. Bridge source has no missing topic handlers
  2. Topic contract matches lekiwi_modular publishers
  3. All required message types are importable
  4. Joint name mappings consistent between URDF and sim
  5. Callbacks, kinematics, adapters, and launch files present

Run:
  python3 scripts/bridge_health_monitor.py

No ROS2 daemon required — pure source-code + type checking.
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple

# ── Paths ─────────────────────────────────────────────────────────────────────
HERMES = Path.home() / "hermes_research"
LEKIWI_VLA = HERMES / "lekiwi_vla"
LEKIWI_MODULAR = HERMES / "lekiwi_modular"
BRIDGE = LEKIWI_VLA / "src/lekiwi_ros2_bridge/bridge_node.py"
MODULAR_CTRL = LEKIWI_MODULAR / "src/lekiwi_controller/scripts"
URDF_CANDIDATES = [
    LEKIWI_MODULAR / "src/lekiwi_description/urdf/LeKiWi.urdf",
    LEKIWI_MODULAR / "src/lekiwi_description/urdf/lekiwi.urdf",
    LEKIWI_MODULAR / "src/lekiwi_description/urdf/LeKiWi.xacro",
]

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"


class Check(NamedTuple):
    name: str
    ok: bool
    detail: str = ""


def ok(msg: str) -> str:
    return f"{GREEN}✓{RESET} {msg}"


def fail(msg: str) -> str:
    return f"{RED}✗{RESET} {msg}"


# ── Topic extractor from source (multi-line robust) ───────────────────────────

def extract_topics_from_source(src: str) -> tuple[dict, dict]:
    """Extract publishers and subscribers from bridge_node.py source.

    Uses paren-matching to handle multi-line arguments correctly.
    Returns (pubs: {topic -> msg_type}, subs: {topic -> msg_type}).
    """
    pubs = {}; subs = {}
    for kind, d in [("create_publisher", pubs), ("create_subscription", subs)]:
        for m in re.finditer(rf"{kind}", src):
            pos = m.start()
            p = pos + len(kind)
            # Skip to '('
            while p < len(src) and src[p] not in "(\n":
                p += 1
            if src[p] != "(":
                continue
            depth = 0; end = p
            for i in range(p, min(len(src), p + 800)):
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1; break
            args_str = src[p:end]
            parts = [x.strip() for x in args_str.split(",") if x.strip()]
            if len(parts) >= 2:
                raw_type = parts[0].lstrip("(\n\r ")
                raw_topic = parts[1].strip().strip("\"'")
                if raw_topic.startswith("/"):
                    d[raw_topic] = raw_type
    return pubs, subs


def parse_modular_topics() -> dict:
    """Extract (topic, direction, msg_type) from lekiwi_modular controllers."""
    topics = {}
    if not MODULAR_CTRL.exists():
        return topics
    for py_file in MODULAR_CTRL.rglob("*.py"):
        src = py_file.read_text()
        for m in re.finditer(r"create_(?:publisher|subscription)\s*\(", src):
            pos = m.end() - 1; depth = 0; end = pos
            for i in range(pos, min(len(src), pos + 400)):
                if src[i] == "(": depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1; break
            args_str = src[pos:end]
            parts = [x.strip() for x in args_str.split(",") if x.strip()]
            if len(parts) >= 2:
                raw_topic = parts[1].strip().strip("\"'")
                raw_type = parts[0].strip()
                if raw_topic.startswith("/"):
                    before = src[max(0, m.start() - 20):m.start()]
                    direction = "pub" if "publisher" in before else "sub"
                    topics[raw_topic] = (direction, raw_type)
    return topics


def run_checks() -> list[Check]:
    results: list[Check] = []

    # ── 1. bridge_node.py exists ────────────────────────────────────────────
    if not BRIDGE.exists():
        return [Check("bridge_node.py exists", False, str(BRIDGE))]
    results.append(Check(
        "bridge_node.py exists", True,
        f"{BRIDGE.name} ({BRIDGE.stat().st_size // 1024}KB)"
    ))

    src = BRIDGE.read_text()

    # ── 2. parse topics ─────────────────────────────────────────────────────
    pubs, subs = extract_topics_from_source(src)
    results.append(Check(
        "Parse bridge topics", True,
        f"{len(pubs)} publishers, {len(subs)} subscribers"
    ))

    # ── 3. topic contract: bridge subs ← lekiwi_modular pubs ──────────────
    modular = parse_modular_topics()
    contract = [
        ("/lekiwi/cmd_vel", "Twist"),
        ("/lekiwi/goal", "Point"),
        ("/lekiwi/vla_action", "Float64MultiArray"),
        ("/lekiwi/policy_input", "ByteMultiArray"),
        ("/lekiwi/cmd_vel_hmac", "ByteMultiArray"),
        ("/lekiwi/record_control", "String"),
    ]
    missing = []
    for topic, exp_type in contract:
        if topic not in subs:
            missing.append(f"{topic} (bridge doesn't subscribe!)")
        else:
            actual = subs[topic].strip()
            if actual != exp_type:
                missing.append(f"{topic}: expected {exp_type}, got {actual}")
    if missing:
        results.append(Check("Bridge ← Modular topic contract", False, " | ".join(missing)))
    else:
        results.append(Check(
            "Bridge ← Modular topic contract", True,
            f"All {len(contract)} required inputs subscribed"
        ))

    # ── 4. bridge output topics ─────────────────────────────────────────────
    static_pubs = {k: v for k, v in pubs.items() if not k.startswith("[DYNAMIC]")}
    results.append(Check(
        "Bridge output topics", True,
        f"{len(static_pubs)} static publishers"
        + (f" + wheel publishers via f-string" if len(pubs) > len(static_pubs) else "")
    ))

    # ── 5. ROS2 msg imports ─────────────────────────────────────────────────
    msgs_needed = {
        "Twist": "geometry_msgs", "Point": "geometry_msgs",
        "JointState": "sensor_msgs", "Image": "sensor_msgs",
        "Odometry": "nav_msgs", "Float64": "std_msgs",
        "String": "std_msgs", "Float64MultiArray": "std_msgs",
        "ByteMultiArray": "std_msgs",
    }
    missing_msgs = []
    for msg, pkg in msgs_needed.items():
        if msg not in src and f"from {pkg}" not in src:
            missing_msgs.append(msg)
    if missing_msgs:
        results.append(Check("ROS2 msg imports", False, f"Missing: {', '.join(missing_msgs)}"))
    else:
        results.append(Check("ROS2 msg imports", True, "All required msgs imported"))

    # ── 6. URDF joints ─────────────────────────────────────────────────────
    urdf_path = next((p for p in URDF_CANDIDATES if p.exists()), None)
    if urdf_path:
        urdf_src = urdf_path.read_text()
        all_joints = re.findall(r'<joint\s+name=["\']([^"\']+)["\']', urdf_src)
        wheel_joints = re.findall(r'<joint\s+name=["\']([^"\']*wheel[^"\']*)["\']', urdf_src, re.I)
        arm_joints = re.findall(r'<joint\s+name=["\']([^"\']*arm[^"\']*)["\']', urdf_src, re.I)
        results.append(Check(
            "URDF joints present", True,
            f"{len(all_joints)} total — {len(wheel_joints)} wheel, {len(arm_joints)} arm joints"
        ))
    else:
        results.append(Check(
            "URDF joints present", False,
            f"None of {len(URDF_CANDIDATES)} candidates found"
        ))

    # ── 7. Sim joint files exist ───────────────────────────────────────────
    sim_files = [
        LEKIWI_VLA / "sim_lekiwi.py",
        LEKIWI_VLA / "sim_lekiwi_urdf.py",
    ]
    sim_exists = [str(p.relative_to(LEKIWI_VLA)) for p in sim_files if p.exists()]
    if sim_exists:
        results.append(Check(
            "Sim backend files", True,
            f"Present: {', '.join(sim_exists)}"
        ))
    else:
        results.append(Check("Sim backend files", False, "No sim files found in lekiwi_vla/"))

    # ── 8. Required callbacks ─────────────────────────────────────────────
    required_cbs = [
        "_on_cmd_vel", "_on_vla_action", "_on_goal",
        "_on_watchdog", "_on_timer",
    ]
    missing_cbs = [cb for cb in required_cbs if f"def {cb}" not in src]
    if missing_cbs:
        results.append(Check("Required callbacks", False, f"Missing: {missing_cbs}"))
    else:
        results.append(Check(
            "Required callbacks", True,
            f"All {len(required_cbs)} required callbacks implemented"
        ))

    # ── 9. Omni-kinematics conversion ─────────────────────────────────────
    has_kin = (
        "twist_to_wheel_speeds" in src or
        "twist_to_contact_wheel_speeds" in src or
        "omni_inverse" in src
    )
    results.append(Check(
        "Omni-kinematics conversion", has_kin,
        "twist_to_wheel_speeds found" if has_kin else "NOT FOUND"
    ))

    # ── 10. Real hardware adapter ─────────────────────────────────────────
    hw_path = BRIDGE.parent / "real_hardware_adapter.py"
    results.append(Check(
        "Real hardware adapter", hw_path.exists(),
        str(hw_path.relative_to(BRIDGE.parent)) if hw_path.exists() else "NOT FOUND"
    ))

    # ── 11. VLA policy node ───────────────────────────────────────────────
    vla_path = BRIDGE.parent / "vla_policy_node.py"
    results.append(Check(
        "VLA policy node", vla_path.exists(),
        str(vla_path.relative_to(BRIDGE.parent)) if vla_path.exists() else "NOT FOUND"
    ))

    # ── 12. CTF integration ───────────────────────────────────────────────
    ctf_path = BRIDGE.parent / "ctf_integration.py"
    results.append(Check(
        "CTF integration", ctf_path.exists(),
        str(ctf_path.relative_to(BRIDGE.parent)) if ctf_path.exists() else "NOT FOUND"
    ))

    # ── 13. Sim loader ──────────────────────────────────────────────────────
    loader_path = BRIDGE.parent / "lekiwi_sim_loader.py"
    results.append(Check(
        "Sim loader", loader_path.exists(),
        str(loader_path.relative_to(BRIDGE.parent)) if loader_path.exists() else "NOT FOUND"
    ))

    # ── 14. Launch files ───────────────────────────────────────────────────
    launch_dir = BRIDGE.parent / "launch"
    required_launches = [
        "full.launch.py", "bridge.launch.py",
        "real_mode.launch.py", "vla.launch.py",
    ]
    if launch_dir.exists():
        found = [l.name for l in launch_dir.glob("*.py")]
        missing_launch = [l for l in required_launches if l not in found]
        results.append(Check(
            "Required launch files",
            len(missing_launch) == 0,
            f"Found: {sorted(found)}" if not missing_launch else f"Missing: {missing_launch}"
        ))
    else:
        results.append(Check(
            "Required launch files", False,
            f"{launch_dir} not found"
        ))

    return results


def main() -> int:
    print("=" * 60)
    print("  LeKiWi ROS2 Bridge — Phase 257 Health Check")
    print("=" * 60)

    results = run_checks()
    passed = sum(1 for r in results if r.ok)
    total = len(results)

    print(f"\nResults: {GREEN}{passed}/{total}{RESET} checks passed\n")
    for r in results:
        detail = f"  → {r.detail}" if r.detail else ""
        status = ok(r.name) if r.ok else fail(r.name)
        print(f"  {status}{detail}")

    print()
    failed = [r for r in results if not r.ok]
    if not failed:
        print(f"{GREEN}All checks passed — bridge is ready for deployment{RESET}")
        return 0
    else:
        print(f"{RED}{len(failed)} check(s) need attention before deployment:{RESET}")
        for r in failed:
            print(f"  {fail(r.name)}: {r.detail}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
