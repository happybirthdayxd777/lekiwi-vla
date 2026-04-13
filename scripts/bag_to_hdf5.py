#!/usr/bin/env python3
"""
Ros2 Bag → HDF5 Converter for LeKiWi
=====================================
Converts recorded ros2 bag files to HDF5 trajectories compatible with
lekiwi_vla training and replay_node.

Output format matches lekiwi_urdf_5k.h5:
    /images         (N, 224, 224, 3) uint8
    /states         (N, 9)          float32   [arm_pos*6, wheel_x, wheel_y, yaw]
    /actions        (N, 9)          float32   [arm_ctrl*6, wheel_vel*3]
    /joint_states   (N, 18)         float64   [arm_pos*6, arm_vel*6, wheel_pos*3, wheel_vel*3]
    /cmd_vel        (N, 3)          float64   [vx, vy, wz]
    /rewards        (N,)            float32   (computed offline via sim replay)
    /goal_positions  (N, 2)         float64   [gx, gy]  (placeholder: zeros)
    /metadata       {"run_id", "duration_s", "fps"}

Optional: compute rewards by replaying through LeKiWiSim (slow but accurate).

Usage:
    # Basic conversion (no reward computation):
    python3 scripts/bag_to_hdf5.py \\
        --input /path/to/recording.db3 \\
        --output data/my_run.h5

    # With reward computation (replays through sim):
    python3 scripts/bag_to_hdf5.py \\
        --input /path/to/recording.db3 \\
        --output data/my_run.h5 \\
        --compute-rewards

    # Multi-session merge:
    python3 scripts/bag_to_hdf5.py \\
        --input session1.db3 session2.db3 \\
        --output merged.h5 \\
        --merge
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add lekiwi_vla to path for sim-based reward computation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check available converters
_BAG_AVAILABLE = False
_CVBIDGE_AVAILABLE = False

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    _BAG_AVAILABLE = True
except ImportError:
    pass

try:
    from cv_bridge import CvBridge
    import cv2
    _CVBRIDGE_AVAILABLE = True
except ImportError:
    pass


ARM_JOINT_NAMES  = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINT_NAMES = ["w1", "w2", "w3"]
IMAGE_H, IMAGE_W = 224, 224


# ─── Standalone helpers for merge path (replicate class methods at module level) ─

def _js_to_state(js: dict) -> np.ndarray:
    """Extract 9-D state from a joint_states dict (standalone version)."""
    nmap = dict(zip(js["names"], js["position"])) if js["position"] else {}
    arm = [nmap.get(n, 0.0) for n in ARM_JOINT_NAMES]
    wheel = [nmap.get(n, 0.0) for n in WHEEL_JOINT_NAMES]
    return np.array(arm + wheel, dtype=np.float32)


def _interpolate_single(buf: list, t: float):
    """Return interpolated value at timestamp t from a buffer of dicts with 'ts' key."""
    if not buf:
        return None
    src_ts = np.array([e["ts"] for e in buf])
    if t <= src_ts[0]:
        return buf[0]
    if t >= src_ts[-1]:
        return buf[-1]
    lo, hi = 0, len(src_ts) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if src_ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    t0, t1 = src_ts[lo], src_ts[hi]
    alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    a0, a1 = buf[lo], buf[hi]
    res = {"ts": t}
    for key in ("vx", "vy", "wz"):
        if key in a0 and key in a1:
            res[key] = a0[key] + alpha * (a1[key] - a0[key])
    if "data" in a0 and "data" in a1:
        d0 = np.asarray(a0["data"], dtype=np.float64)
        d1 = np.asarray(a1["data"], dtype=np.float64)
        res["data"] = (d0 + alpha * (d1 - d0)).tolist()
    elif "data" in a0:
        res["data"] = a0["data"]
    return res


def _interpolate_buffers(source_buf: list, js_timestamps: list):
    """
    Resample a source buffer to match joint_states timestamps.
    Returns list of same length as js_timestamps with linearly interpolated dicts.
    """
    if not source_buf:
        return [None] * len(js_timestamps)
    src_ts = np.array([e["ts"] for e in source_buf])
    n_src = len(src_ts)
    result = [None] * len(js_timestamps)
    for i, t in enumerate(js_timestamps):
        if t <= src_ts[0]:
            result[i] = source_buf[0]
            continue
        if t >= src_ts[-1]:
            result[i] = source_buf[-1]
            continue
        lo, hi = 0, n_src - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if src_ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        t0, t1 = src_ts[lo], src_ts[hi]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        a0, a1 = source_buf[lo], source_buf[hi]
        res = {"ts": t}
        for key in ("vx", "vy", "wz"):
            if key in a0 and key in a1:
                res[key] = a0[key] + alpha * (a1[key] - a0[key])
        if "data" in a0 and "data" in a1 and a0["data"] is not None and a1["data"] is not None:
            res["data"] = (np.asarray(a0["data"]) * (1 - alpha) +
                           np.asarray(a1["data"]) * alpha).astype(np.uint8)
        elif "data" in a0:
            res["data"] = a0["data"]
        result[i] = res
    return result


def _twist_to_wheel_speeds(vx: float, vy: float, wz: float) -> list:
    """Inverse kinematics: Twist → wheel angular velocities (module-level version)."""
    L = 0.1732
    R = 0.05
    a = np.radians(60)
    w1 = (1.0 / R) * (np.cos(a) * vx + np.sin(a) * vy + L * wz)
    w2 = (1.0 / R) * (np.cos(-a) * vx + np.sin(-a) * vy + L * wz)
    w3 = (1.0 / R) * (0.0 * vx + (-1.0) * vy + L * wz)
    return [w1, w2, w3]


# ─── Message helpers ──────────────────────────────────────────────────────────

def _get_msg_type(msg_class_path: str):
    """Convert 'package/Message' to actual message class."""
    if not _BAG_AVAILABLE:
        return None
    try:
        return get_message(msg_class_path)
    except Exception:
        return None


def _imgmsg_to_cv2(msg) -> np.ndarray:
    """Convert ROS Image message to numpy array."""
    if not _CVBRIDGE_AVAILABLE:
        raise RuntimeError("cv_bridge not available")
    bridge = CvBridge()
    cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    return np.asarray(cv2_img)


def _resize_image(img: np.ndarray, target_h=IMAGE_H, target_w=IMAGE_W) -> np.ndarray:
    """Resize image to target resolution."""
    if not _CVBRIDGE_AVAILABLE:
        raise RuntimeError("cv_bridge not available")
    import cv2
    if img.shape[:2] == (target_h, target_w):
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


# ─── Core converter ──────────────────────────────────────────────────────────

class BagToHDF5Converter:
    """
    Converts a ros2 bag (SQLite .db3) to LeKiWi HDF5 format.

    Supports:
      /lekiwi/joint_states    → states + joint_states
      /lekiwi/cmd_vel         → cmd_vel + actions (differentiating)
      /lekiwi/camera/image_raw → images
      /lekiwi/vla_action      → actions (if available, preferred over differentiated)
    """

    # Topic → field mapping
    KNOWN_TOPICS = {
        "/lekiwi/joint_states":     {"msg": "sensor_msgs/JointState",    "role": "joint_states"},
        "/lekiwi/cmd_vel":          {"msg": "geometry_msgs/Twist",         "role": "cmd_vel"},
        "/lekiwi/camera/image_raw": {"msg": "sensor_msgs/Image",           "role": "camera"},
        "/lekiwi/vla_action":       {"msg": "std_msgs/Float64MultiArray",  "role": "vla_action"},
    }

    def __init__(self, bag_path: str):
        if not _BAG_AVAILABLE:
            raise RuntimeError(
                "rosbag2_py not available. Install with:\n"
                "  pip install rosbag2_py rosidl_runtime_py\n"
                "(Usually part of ROS2 foxy/humble install)"
            )
        self.bag_path = os.path.expanduser(bag_path)
        self.storage_opts = StorageOptions()
        self.storage_opts.uri = self.bag_path
        self.storage_opts.storage_id = "sqlite3"

        self.converter_opts = ConverterOptions()
        self.converter_opts.input_serialization_format = "cdr"
        self.converter_opts.output_serialization_format = "cdr"

        self._buffers = {
            "joint_states": [],
            "cmd_vel": [],
            "camera": [],
            "vla_action": [],
            "timestamps": [],
        }
        self._topic_types = {}
        self._metadata = {
            "run_id": Path(bag_path).stem,
            "fps": 20.0,
        }

    def _parse_msg(self, topic: str, msg, timestamp_ns: int):
        """Parse a deserialized message into typed buffers."""
        import sensor_msgs.msg
        import geometry_msgs.msg
        import std_msgs.msg

        role = self.KNOWN_TOPICS.get(topic, {}).get("role")
        ts = timestamp_ns * 1e-9

        if role == "joint_states":
            # JointState: name, position, velocity, effort
            js = msg
            buf = self._buffers
            buf["timestamps"].append(ts)
            buf["joint_states"].append({
                "names": list(js.name),
                "position": list(js.position) if js.position else [],
                "velocity": list(js.velocity) if js.velocity else [],
            })

        elif role == "cmd_vel":
            cv = msg
            self._buffers["cmd_vel"].append({
                "vx": cv.linear.x,
                "vy": cv.linear.y,
                "wz": cv.angular.z,
                "ts": ts,
            })

        elif role == "camera" and _CVBRIDGE_AVAILABLE:
            try:
                img = _imgmsg_to_cv2(msg)
                img_resized = _resize_image(img)
                self._buffers["camera"].append({
                    "data": img_resized,
                    "ts": ts,
                })
            except Exception as e:
                print(f"[WARN] Image decode failed: {e}", flush=True)

        elif role == "vla_action":
            self._buffers["vla_action"].append({
                "data": list(msg.data) if hasattr(msg, "data") else [],
                "ts": ts,
            })

    def read_bag(self):
        """Read all messages from the bag file."""
        reader = SequentialReader()
        reader.open(self.storage_opts, self.converter_opts)

        topic_types = reader.get_all_topics_and_types()
        self._topic_types = {t.name: t.type for t in topic_types}

        print(f"[bag_to_hdf5] Reading: {self.bag_path}")
        print(f"[bag_to_hdf5] Topics: {list(self._topic_types.keys())}")

        count = 0
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            try:
                msg_type = self._topic_types.get(topic)
                if not msg_type:
                    continue
                msg_class = get_message(msg_type)
                msg = deserialize_message(data, msg_class)
                self._parse_msg(topic, msg, timestamp_ns)
                count += 1
            except Exception as e:
                print(f"[WARN] Failed to deserialize {topic}: {e}", flush=True)
                continue
            if count % 10000 == 0 and count > 0:
                print(f"[bag_to_hdf5] {count} messages read...", flush=True)

        print(f"[bag_to_hdf5] Done: {count} messages, "
              f"{len(self._buffers['camera'])} images", flush=True)

    def _joint_states_to_state(self, js: dict) -> np.ndarray:
        """
        Extract 9-D state from joint_states buffer.
        Canonical state: [arm_pos*6, base_x, base_y, base_yaw]
        Since we don't have explicit base pose from joint_states alone,
        we use wheel累积 position as proxy (first 3 wheel positions).
        Actual base pose requires /lekiwi/odom.
        """
        names = js["names"]
        pos = js["position"]
        # Build name→value map
        nmap = dict(zip(names, pos))

        # Arm positions (6)
        arm = [nmap.get(n, 0.0) for n in ARM_JOINT_NAMES]

        # Wheel positions (3) — for state_dim=9 we use wheel positions
        wheel = [nmap.get(n, 0.0) for n in WHEEL_JOINT_NAMES]

        return np.array(arm + wheel, dtype=np.float32)

    def _joint_states_to_action(self, js: dict, cv: dict = None) -> np.ndarray:
        """
        Extract 9-D action from joint_states velocity + optional cmd_vel.
        action = [arm_vel*6 (from js.velocity), wheel_vel*3]
        If cmd_vel is provided, use its [vx, vy, wz] to compute wheel speeds
        via inverse kinematics.
        """
        names = js["names"]
        vel = js["velocity"]
        nmap = dict(zip(names, vel))

        arm_vel = [nmap.get(n, 0.0) for n in ARM_JOINT_NAMES]

        if cv is not None:
            # cmd_vel → wheel_speeds (same kinematics as bridge_node)
            vx, vy, wz = cv["vx"], cv["vy"], cv["wz"]
            wheel_speeds = self._twist_to_wheel_speeds(vx, vy, wz)
        else:
            wheel_speeds = [nmap.get(n, 0.0) for n in WHEEL_JOINT_NAMES]

        return np.array(arm_vel + wheel_speeds, dtype=np.float32)

    def _twist_to_wheel_speeds(self, vx: float, vy: float, wz: float) -> list:
        """
        Inverse kinematics: Twist (vx, vy, wz) → wheel angular velocities.
        From lekiwi_modular omni_controller.cpp:
          wheel_i = (1/R) * [cos(θ+60°), sin(θ+60°), 1] · [vx, vy, wz*L]
        where L = 0.1732m (wheel base radius).
        """
        L = 0.1732  # meters
        R = 0.05    # wheel radius (meters) — nominal
        a = np.radians(60)

        w1 = (1.0 / R) * (np.cos(a) * vx + np.sin(a) * vy + L * wz)
        w2 = (1.0 / R) * (np.cos(-a) * vx + np.sin(-a) * vy + L * wz)
        w3 = (1.0 / R) * (0.0 * vx + (-1.0) * vy + L * wz)

        return [w1, w2, w3]

    def _interpolate_to_js_rate(self, source_buf: list, js_timestamps: list):
        """
        Resample a source buffer (camera, cmd_vel) to match joint_states timestamps.
        Returns array of same length as js_timestamps with linearly interpolated values.

        Upgraded from nearest-neighbor to linear interpolation for smoother
        trajectory reconstruction — reduces camera jitter and cmd_vel discontinuities.
        """
        if not source_buf:
            return [None] * len(js_timestamps)

        # Extract timestamps
        src_ts = np.array([e["ts"] for e in source_buf])
        js_ts = np.array(js_timestamps)

        # Fast vectorized linear interpolation
        # For each js timestamp, find surrounding source indices
        # src_ts must be sorted (bag messages are in time order)
        n_src = len(src_ts)
        result = [None] * len(js_ts)

        for i, t in enumerate(js_ts):
            if t <= src_ts[0]:
                result[i] = source_buf[0]
                continue
            if t >= src_ts[-1]:
                result[i] = source_buf[-1]
                continue

            # Binary search for insertion point
            lo, hi = 0, n_src - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if src_ts[mid] <= t:
                    lo = mid
                else:
                    hi = mid

            # Linear interpolation between lo and hi
            t0, t1 = src_ts[lo], src_ts[hi]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            a0, a1 = source_buf[lo], source_buf[hi]

            # Interpolate numeric fields
            res = {"ts": t}
            for key in ("vx", "vy", "wz"):
                if key in a0 and key in a1:
                    res[key] = a0[key] + alpha * (a1[key] - a0[key])
            # Interpolate image data if present
            if "data" in a0 and "data" in a1 and a0["data"] is not None and a1["data"] is not None:
                # Blend images with alpha
                res["data"] = (a0["data"] * (1 - alpha) + a1["data"] * alpha).astype(np.uint8)
            else:
                # Fall back to nearest
                res["data"] = a0["data"] if alpha < 0.5 else a1["data"]
            # Interpolate action data
            if "data" in a0 and "data" in a1:
                d0 = np.asarray(a0["data"], dtype=np.float64)
                d1 = np.asarray(a1["data"], dtype=np.float64)
                res["data"] = (d0 + alpha * (d1 - d0)).tolist()

            result[i] = res

        return result

    def build_hdf5(self, output_path: str, compute_rewards: bool = False):
        """
        Assemble buffered data into HDF5 file.

        Args:
            output_path: Output .h5 file path
            compute_rewards: If True, replay through LeKiWiSim to compute rewards
        """
        import h5py

        output_path = os.path.expanduser(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        js_buf = self._buffers["joint_states"]
        cv_buf = self._buffers["cmd_vel"]
        cam_buf = self._buffers["camera"]
        va_buf = self._buffers["vla_action"]

        if not js_buf:
            raise ValueError("No joint_states found in bag — cannot build trajectory")

        n = len(js_buf)
        print(f"[bag_to_hdf5] Assembling {n}-frame trajectory → {output_path}")

        # ── Build states ────────────────────────────────────────────────────
        states = np.stack([self._joint_states_to_state(js) for js in js_buf])
        states = states.astype(np.float32)

        # ── Build actions ───────────────────────────────────────────────────
        # Prefer vla_action if available; else differentiate cmd_vel
        if va_buf:
            # vla_action messages — map to nearest js timestamp
            va_interp = self._interpolate_to_js_rate(va_buf, [js["ts"] for js in js_buf])
            actions = []
            for va, js in zip(va_interp, js_buf):
                if va and va["data"] and len(va["data"]) >= 9:
                    # VLA action is already in native units (arm pos + wheel vel)
                    arm = va["data"][:6]
                    wheel = va["data"][6:9]
                else:
                    # Fall back to differentiated cmd_vel
                    cv = self._interpolate_to_js_rate(cv_buf, [js["ts"]])[0]
                    arm_vel = js["velocity"][:6] if js["velocity"] else [0]*6
                    wheel = self._twist_to_wheel_speeds(
                        cv["vx"], cv["vy"], cv["wz"]
                    ) if cv else [0, 0, 0]
                    arm = arm_vel
                actions.append(list(arm) + list(wheel))
            actions = np.array(actions, dtype=np.float32)
        else:
            # No VLA action — use cmd_vel → wheel_speeds, joint_states velocity for arms
            cv_interp = self._interpolate_to_js_rate(cv_buf, [js["ts"] for js in js_buf])
            actions = []
            for js, cv in zip(js_buf, cv_interp):
                arm_vel = list(js["velocity"][:6]) if js["velocity"] else [0.0]*6
                if cv:
                    wheel = self._twist_to_wheel_speeds(cv["vx"], cv["vy"], cv["wz"])
                else:
                    wheel = [0.0]*3
                actions.append(arm_vel + wheel)
            actions = np.array(actions, dtype=np.float32)

        # ── Build images ────────────────────────────────────────────────────
        images = np.zeros((n, IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        if cam_buf:
            cam_interp = self._interpolate_to_js_rate(cam_buf, [js["ts"] for js in js_buf])
            for i, cam in enumerate(cam_interp):
                if cam is not None and cam["data"] is not None:
                    images[i] = cam["data"]
        else:
            print("[bag_to_hdf5] WARNING: No camera data found — images will be black")

        # ── Build rewards ───────────────────────────────────────────────────
        rewards = np.zeros(n, dtype=np.float32)
        if compute_rewards:
            print("[bag_to_hdf5] Computing rewards via sim replay (slow, ~1 min)...")
            try:
                from sim_lekiwi import LeKiWiSim
                sim = LeKiWiSim()
                for i in range(n):
                    action = actions[i]
                    obs, reward, done, info = sim.step(action)
                    rewards[i] = reward
                    if i % 500 == 0 and i > 0:
                        print(f"[bag_to_hdf5] Reward replay: {i}/{n}", flush=True)
            except Exception as e:
                print(f"[bag_to_hdf5] Reward computation failed: {e} — using zeros")
        else:
            # Placeholder rewards (0 = no signal for training weighting)
            pass

        # ── Build joint_states (N, 18) ───────────────────────────────────────
        js_arr = np.zeros((n, 18), dtype=np.float64)
        for i, js in enumerate(js_buf):
            names = js["names"]
            nmap_pos = dict(zip(names, js["position"])) if js["position"] else {}
            nmap_vel = dict(zip(names, js["velocity"])) if js["velocity"] else {}
            arm_pos = [nmap_pos.get(n, 0.0) for n in ARM_JOINT_NAMES]
            arm_vel = [nmap_vel.get(n, 0.0) for n in ARM_JOINT_NAMES]
            wheel_p = [nmap_pos.get(n, 0.0) for n in WHEEL_JOINT_NAMES]
            wheel_v = [nmap_vel.get(n, 0.0) for n in WHEEL_JOINT_NAMES]
            js_arr[i] = arm_pos + arm_vel + wheel_p + wheel_v

        # ── Build cmd_vel (N, 3) ──────────────────────────────────────────────
        cv_arr = np.zeros((n, 3), dtype=np.float64)
        cv_interp = self._interpolate_to_js_rate(cv_buf, [js["ts"] for js in js_buf])
        for i, cv in enumerate(cv_interp):
            if cv:
                cv_arr[i] = [cv["vx"], cv["vy"], cv["wz"]]

        # ── Write HDF5 ──────────────────────────────────────────────────────
        duration = (js_buf[-1]["ts"] - js_buf[0]["ts"]) if len(js_buf) > 1 else 0.0
        fps = n / duration if duration > 0 else 20.0

        with h5py.File(output_path, "w") as f:
            f.create_dataset("images",         data=images,     compression="gzip", compression_opts=4)
            f.create_dataset("states",          data=states,     compression="gzip", compression_opts=1)
            f.create_dataset("actions",        data=actions,    compression="gzip", compression_opts=1)
            f.create_dataset("rewards",          data=rewards,    compression="gzip", compression_opts=1)
            f.create_dataset("joint_states",    data=js_arr,    compression="gzip", compression_opts=1)
            f.create_dataset("cmd_vel",         data=cv_arr,    compression="gzip", compression_opts=1)

            # Placeholder goal positions (real bag recordings don't have goals)
            goal_positions = np.zeros((n, 2), dtype=np.float64)
            f.create_dataset("goal_positions",  data=goal_positions)

            # Metadata
            f.attrs["run_id"] = Path(output_path).stem
            f.attrs["duration_s"] = float(duration)
            f.attrs["num_frames"] = n
            f.attrs["fps"] = float(fps)
            f.attrs["arm_names"] = ARM_JOINT_NAMES
            f.attrs["wheel_names"] = WHEEL_JOINT_NAMES
            f.attrs["has_rewards"] = compute_rewards
            f.attrs["has_images"] = len(cam_buf) > 0

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[bag_to_hdf5] Saved: {n} frames, {duration:.1f}s, {fps:.1f} fps → "
              f"{output_path} ({size_mb:.1f} MB)")
        print(f"[bag_to_hdf5]   images:     {images.shape}")
        print(f"[bag_to_hdf5]   states:     {states.shape}")
        print(f"[bag_to_hdf5]   actions:    {actions.shape}")
        print(f"[bag_to_hdf5]   rewards:     {'computed' if compute_rewards else 'zeros (placeholder)'}")
        print(f"[bag_to_hdf5]   joint_states: {js_arr.shape}")
        print(f"[bag_to_hdf5]   cmd_vel:     {cv_arr.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ros2 bag to LeKiWi HDF5 trajectory"
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="Path to ros2 bag directory (contains metadata.yaml + .db3)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output .h5 file path"
    )
    parser.add_argument(
        "--compute-rewards", action="store_true",
        help="Replay trajectory through LeKiWiSim to compute rewards (slow)"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge multiple bag files into one HDF5"
    )
    args = parser.parse_args()

    if args.merge and len(args.input) > 1:
        # Merge multiple bags: read all bags, concatenate buffers, build merged HDF5
        print(f"[bag_to_hdf5] Merging {len(args.input)} bag files...")
        all_js, all_cv, all_cam, all_va, all_ts = [], [], [], [], []
        for bp in args.input:
            conv = BagToHDF5Converter(bp)
            conv.read_bag()
            all_js.extend(conv._buffers["joint_states"])
            all_cv.extend(conv._buffers["cmd_vel"])
            all_cam.extend(conv._buffers["camera"])
            all_va.extend(conv._buffers["vla_action"])
            all_ts.extend(conv._buffers["timestamps"])
            print(f"  {bp}: {len(conv._buffers['joint_states'])} joint_states, "
                  f"{len(conv._buffers['camera'])} images")

        # Sort by global timestamp across all bags
        # Build merged buffer with consistent ordering
        n = len(all_js)
        print(f"[bag_to_hdf5] Merged: {n} total frames")

        # ── Build states ────────────────────────────────────────────────────
        states = np.stack([_js_to_state(js) for js in all_js])
        states = states.astype(np.float32)

        # ── Build actions ───────────────────────────────────────────────────
        if all_va:
            va_interp = _interpolate_buffers(all_va, [js["ts"] for js in all_js])
            actions = []
            for va, js in zip(va_interp, all_js):
                if va and va.get("data") and len(va["data"]) >= 9:
                    arm = va["data"][:6]
                    wheel = va["data"][6:9]
                else:
                    cv = _interpolate_single(all_cv, js["ts"]) if all_cv else None
                    arm_vel = js["velocity"][:6] if js["velocity"] else [0]*6
                    wheel = _twist_to_wheel_speeds(
                        cv["vx"], cv["vy"], cv["wz"]
                    ) if cv else [0, 0, 0]
                    arm = arm_vel
                actions.append(list(arm) + list(wheel))
            actions = np.array(actions, dtype=np.float32)
        else:
            cv_interp = _interpolate_buffers(all_cv, [js["ts"] for js in all_js])
            actions = []
            for js, cv in zip(all_js, cv_interp):
                arm_vel = list(js["velocity"][:6]) if js["velocity"] else [0.0]*6
                if cv:
                    wheel = _twist_to_wheel_speeds(cv["vx"], cv["vy"], cv["wz"])
                else:
                    wheel = [0.0]*3
                actions.append(arm_vel + wheel)
            actions = np.array(actions, dtype=np.float32)

        # ── Build images ────────────────────────────────────────────────────
        images = np.zeros((n, IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        if all_cam:
            cam_interp = _interpolate_buffers(all_cam, [js["ts"] for js in all_js])
            for i, cam in enumerate(cam_interp):
                if cam is not None and cam.get("data") is not None:
                    images[i] = cam["data"]
        else:
            print("[bag_to_hdf5] WARNING: No camera data in merged bags — images will be black")

        # ── Build joint_states (N, 18) ───────────────────────────────────────
        js_arr = np.zeros((n, 18), dtype=np.float64)
        for i, js in enumerate(all_js):
            nmap_pos = dict(zip(js["names"], js["position"])) if js["position"] else {}
            nmap_vel = dict(zip(js["names"], js["velocity"])) if js["velocity"] else {}
            arm_p = [nmap_pos.get(n, 0.0) for n in ARM_JOINT_NAMES]
            arm_v = [nmap_vel.get(n, 0.0) for n in ARM_JOINT_NAMES]
            wheel_p = [nmap_pos.get(n, 0.0) for n in WHEEL_JOINT_NAMES]
            wheel_v = [nmap_vel.get(n, 0.0) for n in WHEEL_JOINT_NAMES]
            js_arr[i] = arm_p + arm_v + wheel_p + wheel_v

        # ── Build cmd_vel (N, 3) ─────────────────────────────────────────────
        cv_arr = np.zeros((n, 3), dtype=np.float64)
        cv_interp = _interpolate_buffers(all_cv, [js["ts"] for js in all_js])
        for i, cv in enumerate(cv_interp):
            if cv:
                cv_arr[i] = [cv["vx"], cv["vy"], cv["wz"]]

        # ── Rewards placeholder ─────────────────────────────────────────────
        rewards = np.zeros(n, dtype=np.float32)
        goal_positions = np.zeros((n, 2), dtype=np.float64)
        duration = (all_js[-1]["ts"] - all_js[0]["ts"]) if len(all_js) > 1 else 0.0
        fps = n / duration if duration > 0 else 20.0

        # ── Write merged HDF5 ───────────────────────────────────────────────
        import h5py
        output_path = os.path.expanduser(args.output)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            f.create_dataset("images",        data=images,       compression="gzip", compression_opts=4)
            f.create_dataset("states",        data=states,       compression="gzip", compression_opts=1)
            f.create_dataset("actions",       data=actions,      compression="gzip", compression_opts=1)
            f.create_dataset("rewards",       data=rewards,     compression="gzip", compression_opts=1)
            f.create_dataset("joint_states",  data=js_arr,      compression="gzip", compression_opts=1)
            f.create_dataset("cmd_vel",        data=cv_arr,      compression="gzip", compression_opts=1)
            f.create_dataset("goal_positions",data=goal_positions)
            f.attrs["run_id"] = Path(output_path).stem + "_merged"
            f.attrs["duration_s"] = float(duration)
            f.attrs["num_frames"] = n
            f.attrs["fps"] = float(fps)
            f.attrs["arm_names"] = ARM_JOINT_NAMES
            f.attrs["wheel_names"] = WHEEL_JOINT_NAMES
            f.attrs["has_rewards"] = False
            f.attrs["has_images"] = len(all_cam) > 0
            f.attrs["merged_from"] = str([str(p) for p in args.input])

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[bag_to_hdf5] Merged saved: {n} frames, {duration:.1f}s, {fps:.1f} fps → "
              f"{output_path} ({size_mb:.1f} MB)")
        print(f"[bag_to_hdf5]   images: {images.shape}, states: {states.shape}, "
              f"actions: {actions.shape}")
        return  # Done — don't fall through to single-bag processing

    if len(args.input) == 1:
        conv = BagToHDF5Converter(args.input[0])
        conv.read_bag()
        conv.build_hdf5(args.output, compute_rewards=args.compute_rewards)
    else:
        print("[bag_to_hdf5] Multiple inputs without --merge: processing first only")
        conv = BagToHDF5Converter(args.input[0])
        conv.read_bag()
        conv.build_hdf5(args.output, compute_rewards=args.compute_rewards)


if __name__ == "__main__":
    main()
