#!/usr/bin/env python3
"""
lekiwi_sim_loader — Unified simulation factory for the ROS2 bridge

Supports three backends:
  sim_type=primitive → LeKiwiSim (fast primitive geometry, no STL meshes)
  sim_type=urdf      → LeKiWiSimURDF (STL meshes from lekiwi_modular URDF)
  mode=real           → RealHardwareAdapter (serial bus, no MuJoCo)

Usage (from bridge_node main()):
    from lekiwi_sim_loader import load_lekiwi_sim
    sim = load_lekiwi_sim(sim_type='urdf', mode='sim', render=False)
"""
import os
import sys

# lekiwi_vla root — where sim_lekiwi.py / sim_lekiwi_urdf.py live
_LEKIWI_VLA = os.path.expanduser("~/hermes_research/lekiwi_vla")
if _LEKIWI_VLA not in sys.path:
    sys.path.insert(0, _LEKIWI_VLA)


def load_lekiwi_sim(sim_type: str = 'urdf', mode: str = 'sim', render: bool = False):
    """
    Factory function returning the appropriate simulation / hardware backend.

    Parameters
    ----------
    sim_type : str
        'primitive'  → LeKiwiSim (fast, no meshes)
        'urdf'       → LeKiWiSimURDF (STL meshes, front + wrist cameras)
    mode : str
        'sim'  → MuJoCo simulation (normal)
        'real' → RealHardwareAdapter (serial bus, no MuJoCo)
    render : bool
        True to enable MuJoCo passive viewer (human inspection mode)

    Returns
    -------
    Backend object with a compatible interface:
        - step(action: np.ndarray) → dict (observation)
        - render(width, height)   → PIL.Image
        - _jpos_idx, _jvel_idx    → dict[str, int]
    """
    if mode == 'real':
        from lekiwi_ros2_bridge.real_hardware_adapter import RealHardwareAdapter
        return RealHardwareAdapter()

    if sim_type == 'primitive':
        from sim_lekiwi import LeKiwiSim
        sim = LeKiwiSim()
        if render:
            _launch_viewer(sim)
        return sim

    # Default: URDF with STL meshes
    from sim_lekiwi_urdf import LeKiWiSimURDF
    sim = LeKiWiSimURDF(render_mode='human' if render else None)
    return sim


def _launch_viewer(sim):
    """Launch MuJoCo passive viewer (non-blocking)."""
    import mujoco.viewer
    import threading

    def _view():
        with mujoco.viewer.passive(sim.model, sim.data) as v:
            while v.is_running():
                v.sync()

    t = threading.Thread(target=_view, daemon=True)
    t.start()
    return t
