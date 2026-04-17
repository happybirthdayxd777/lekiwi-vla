"""
lekiwi_sim_loader — Factory + adapter for LeKiWi simulation backends.
======================================================================
Provides a unified LeKiWiSimBase interface matching sim_lekiwi.py for bridge_node.py.

Backends:
  1. LeKiWiSimDirect  — wraps sim_lekiwi.LeKiwiSim with set_action()+step() pattern
  2. LeKiWiSimWrapper  — wraps sim_lekiwi_urdf.LeKiWiSimURDF with same interface

LeKiWiSim (sim_lekiwi.py) interface:
  step(action[9])     → dict with arm/wheel/base state
  _obs()              → dict
  render() / reset() / launch_viewer()

bridge_node.py expects (LeKiWiSim-style):
  set_action(action[9])   — store action
  step()                  — advance with stored action
  get_state()             → dict
  get_base_pose()         → (x, y, yaw)
"""

import numpy as np
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sim_lekiwi import LeKiwiSim   # note: LeKiwiSim (small i)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LeKiWiSimDirect — adapter for primitive cylinder model
# ─────────────────────────────────────────────────────────────────────────────

class LeKiWiSimDirect:
    """
    Adapter: makes sim_lekiwi.LeKiwiSim compatible with bridge_node interface.

    LeKiwiSim.step(action) → returns dict (Gymnasium-style)
    LeKiWiSim._obs()       → returns dict (raw sensor)

    This adapter:
      set_action(a)  → stores a, calls nothing
      step()         → calls .step(stored_action)
      get_state()    → aliases ._obs()
      get_base_pose()→ extracts (x,y,yaw) from _obs()
    """

    def __init__(self):
        self._sim = LeKiwiSim()
        self._action = np.zeros(9)

    def set_action(self, action: np.ndarray):
        self._action = np.asarray(action, dtype=np.float64).copy()

    def step(self, action: Optional[np.ndarray] = None):
        if action is not None:
            self._action = np.asarray(action, dtype=np.float64).copy()
        # LeKiwiSim.step() returns (obs, reward, done, trunc, info) — ignore extra values
        self._sim.step(self._action)
        return self._sim._obs()

    def get_state(self) -> dict:
        """Return raw observation dict matching LeKiWiSim._obs() keys."""
        return self._sim._obs()

    def get_base_pose(self) -> tuple[float, float, float]:
        obs = self._sim._obs()
        pos = obs["base_position"]   # [x, y, z]
        quat = obs["base_quaternion"]  # [qx, qy, qz, qw]
        yaw = np.arctan2(
            2.0*(quat[3]*quat[2] + quat[0]*quat[1]),
            1.0 - 2.0*(quat[1]**2 + quat[2]**2)
        )
        return float(pos[0]), float(pos[1]), float(yaw)

    def reset(self):
        self._sim.reset()
        self._action = np.zeros(9)

    def render(self, width=640, height=480):
        return self._sim.render(width, height)

    def launch_viewer(self):
        return self._sim.open_viewer()

    @property
    def action_dim(self) -> int:
        return self._sim.action_dim

    def __repr__(self):
        return f"LeKiWiSimDirect({self._sim})"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LeKiWiSimWrapper — adapter for STL-mesh URDF loader
# ─────────────────────────────────────────────────────────────────────────────

class LeKiWiSimWrapper:
    """
    Adapter: makes sim_lekiwi_urdf.LeKiWiSimURDF compatible with bridge_node interface.

    sim_lekiwi_urdf.LeKiWiSimURDF uses:
      step(action[9])  — action[0:6]=arm_torque_norm, action[6:9]=wheel_torque_norm
      _obs()          — dict with different keys

    This adapter:
      set_action(arm[6]+wheel[3])  — arm: position targets, wheel: velocity commands
                                     PD transform → torque → step()
      step()                       — advance with stored action
      get_state()                  → dict matching LeKiWiSim._obs() keys
      get_base_pose()              → (x, y, yaw)
    """

    def __init__(self):
        from sim_lekiwi_urdf import LeKiWiSimURDF as _URDFSim
        self._urdf: _URDFSim = _URDFSim()
        self._action = np.zeros(9)

    def set_action(self, action: np.ndarray):
        """Store arm position targets [6] + wheel velocity targets [3]."""
        self._action = np.asarray(action, dtype=np.float64).copy()

    def step(self, action: Optional[np.ndarray] = None):
        if action is not None:
            self._action = np.asarray(action, dtype=np.float64).copy()
        arm_targets   = self._action[0:6]
        wheel_speeds  = self._action[6:9]

        # Current state
        obs = self._urdf._obs()
        arm_pos = obs["arm_positions"]    # (6,)
        arm_vel = obs["arm_velocities"]  # (6,)
        wheel_vel = obs["wheel_velocities"]  # (3,)

        # PD transforms → normalized torque actions (URDF format)
        kp_arm, kd_arm = 20.0, 5.0
        kp_w,  kd_w    = 5.0,  1.0

        arm_torque = kp_arm*(arm_targets - arm_pos) - kd_arm*arm_vel
        arm_torque_norm = np.clip(arm_torque / 3.14, -1.0, 1.0)

        wheel_torque = kp_w*(wheel_speeds - wheel_vel) - kd_w*wheel_vel
        wheel_torque_norm = np.clip(wheel_torque / 10.0, -0.5, 0.5)

        self._urdf.step(np.concatenate([arm_torque_norm, wheel_torque_norm]))

    def get_state(self) -> dict:
        obs = self._urdf._obs()
        return {
            "arm_positions":    obs["arm_positions"].copy(),
            "arm_velocities":    obs["arm_velocities"].copy(),
            "wheel_positions":   obs.get("wheel_positions", np.zeros(3)),
            "wheel_velocities":  obs["wheel_velocities"].copy(),
            "base_position":     obs["base_position"].copy(),
            "base_quaternion":   obs["base_quaternion"].copy(),
            "base_linear_velocity":  obs["base_linear_velocity"].copy(),
            "base_angular_velocity":obs["base_angular_velocity"].copy(),
        }

    def get_base_pose(self) -> tuple[float, float, float]:
        obs = self._urdf._obs()
        pos  = obs["base_position"]
        quat = obs["base_quaternion"]
        yaw = np.arctan2(
            2.0*(quat[3]*quat[2] + quat[0]*quat[1]),
            1.0 - 2.0*(quat[1]**2 + quat[2]**2)
        )
        return float(pos[0]), float(pos[1]), float(yaw)

    def reset(self):
        self._urdf.reset()
        self._action = np.zeros(9)

    def render(self, width=640, height=480):
        # sim_lekiwi_urdf.render() returns numpy array — convert to PIL for bridge
        arr = self._urdf.render()
        if hasattr(arr, 'convert'):
            return arr   # already PIL
        from PIL import Image
        return Image.fromarray(arr)

    def launch_viewer(self):
        return self._urdf.launch_viewer()

    def __repr__(self):
        return f"LeKiWiSimWrapper(urdf={self._urdf})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_sim(sim_type: str = "primitive", **kwargs):
    """
    Create a LeKiWi simulation backend matching the LeKiWiSim interface.

    Args:
        sim_type: "primitive" → LeKiWiSimDirect (cylinder model, fast+stable)
                  "urdf"       → LeKiWiSimWrapper (STL mesh, real geometry)
        **kwargs: passed to underlying simulator

    Returns an object with:
        set_action(action[9])   arm: position targets, wheel: velocity commands
        step(action?)            advance physics
        get_state()              dict of arm/wheel/base state
        get_base_pose()          (x, y, yaw)
        reset() / render() / launch_viewer()
    """
    if sim_type == "primitive":
        return LeKiWiSimDirect(**kwargs)
    elif sim_type == "urdf":
        return LeKiWiSimWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown sim_type={sim_type!r}. Use 'primitive' or 'urdf'.")
