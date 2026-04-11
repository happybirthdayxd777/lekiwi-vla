#!/usr/bin/env python3
"""
Hardware Interface for LeKiwi Robot
Wraps LeRobot's LeKiwi robot and provides high-level control API.
"""

import time
import numpy as np
from typing import Optional

import sys
sys.path.insert(0, "/Users/i_am_ai/lerobot/src")

from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.types import RobotObservation


class HardwareInterface:
    """
    High-level hardware interface for LeKiwi robot.
    Wraps LeRobot's LeKiwi implementation with action-space mapping.
    """
    
    def __init__(self, port: str = "/dev/ttyACM0", robot_id: str = "lekiwi_client"):
        self.port = port
        self.robot_id = robot_id
        self.robot: Optional[LeKiwi] = None
        self.connected = False
        
        # Action space: 6 arm joints + 3 base wheels = 9 DOF
        self.action_dim = 9
        
    def connect(self, calibrate: bool = False):
        """Connect to the LeKiwi robot via USB."""
        config = LeKiwiConfig(
            port=self.port,
            use_degrees=False,
        )
        
        self.robot = LeKiwi(config)
        self.robot.connect(calibrate=calibrate)
        self.connected = True
        print(f"Connected to LeKiwi on {self.port}")
        
    def disconnect(self):
        """Disconnect from robot."""
        if self.robot:
            self.robot.disconnect()
            self.connected = False
            
    def get_observation(self) -> dict:
        """Get current robot state and camera images."""
        if not self.connected:
            raise RuntimeError("Not connected to robot")
            
        obs = self.robot.capture_observation()
        
        return {
            "arm_positions": [obs[f"arm_{n}_pos"] for n in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]],
            "base_velocities": [obs["x.vel"], obs["y.vel"], obs["theta.vel"]],
            "cameras": {
                "front": obs.get("front", None),
                "wrist": obs.get("wrist", None),
            },
            "timestamp": time.time(),
        }
        
    def send_action(self, action: np.ndarray):
        """
        Send joint position targets to robot.
        
        Args:
            action: array of 9 values [arm_joints(6), base_velocities(3)]
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")
            
        if len(action) != self.action_dim:
            raise ValueError(f"Action must have {self.action_dim} dimensions, got {len(action)}")
            
        # Map action to robot command format
        arm_targets = action[:6]
        base_targets = action[6:9]
        
        self.robot.send_action(arm_targets, base_targets)
        
    def move_base(self, vx: float, vy: float, wz: float):
        """
        Direct base velocity command (useful for explorer mode).
        
        Args:
            vx: forward velocity (m/s)
            vy: sideways velocity (m/s)
            wz: angular velocity (rad/s)
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")
            
        # Map to wheel velocities
        wheel_radius = 0.05
        wheel_base = 0.1732
        
        # Simple differential drive mapping for 3 omni wheels
        wheel_speeds = [
            vx - wz * wheel_base,
            vy + wz * wheel_base * 0.5,
            -vx - wz * wheel_base,
        ]
        
        # Send as velocity targets
        action = [0.0] * 6 + wheel_speeds
        self.send_action(np.array(action))
        
    def stop(self):
        """Emergency stop - zero all velocities."""
        self.send_action(np.zeros(9))
        
    def __repr__(self):
        status = "connected" if self.connected else "disconnected"
        return f"HardwareInterface({self.port}, {status})"


# Test code
if __name__ == "__main__":
    import sys
    
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    
    hw = HardwareInterface(port)
    
    print("LeKiwi Hardware Interface initialized")
    print(f"Port: {port}")
    print(f"Action dim: {hw.action_dim}")
    
    # Try to connect (will fail gracefully on macOS without real robot)
    try:
        hw.connect(calibrate=False)
    except Exception as e:
        print(f"Note: Cannot connect to real robot (expected on development machine): {e}")