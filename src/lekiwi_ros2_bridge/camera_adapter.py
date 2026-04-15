#!/usr/bin/env python3
"""
LeKiWi Camera Adapter — MuJoCo Image → ROS2 Image Bridge
==========================================================
Converts rendered images from LeKiWiSimURDF into ROS2 Image messages.

Supports two cameras:
  - Front camera: 640x480, 20 Hz, used for VLA perception
  - Wrist camera: 640x480, 20 Hz, arm-tip view for precision tasks

The adapter maintains its own render loop to avoid blocking the main
bridge step loop. Images are cached and published at 20 Hz.

Topics:
  /lekiwi/camera/image_raw       (sensor_msgs/Image) — front camera
  /lekiwi/wrist_camera/image_raw (sensor_msgs/Image) — wrist camera

Usage:
  from camera_adapter import CameraAdapter
  adapter = CameraAdapter(sim, node)  # sim=LeKiWiSimURDF instance, node=rclpy.Node
  adapter.start()   # starts background render thread
  adapter.stop()     # stops and joins thread
"""

import threading
import time
from typing import Optional

import numpy as np
from cv_bridge import CvBridge

import rclpy
from sensor_msgs.msg import Image

# ── Camera Configuration ───────────────────────────────────────────────────────
FRONT_CAMERA_NAME = "front"
WRIST_CAMERA_NAME = "wrist"
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 20
CAMERA_QUALITY = 85  # JPEG quality (0-100)


class CameraAdapter:
    """
    Background camera renderer for LeKiWiSimURDF → ROS2 Image bridge.
    
    Maintains a dedicated rendering thread that:
      1. Renders front + wrist cameras from MuJoCo at 20 Hz
      2. Converts to ROS2 Image messages via cv_bridge
      3. Publishes to /lekiwi/camera/image_raw and /lekiwi/wrist_camera/image_raw
    
    Args:
        sim: LeKiWiSimURDF instance (must have .render() and .render_wrist())
        node: rclpy.node.Node instance for publishing
        front_topic: ROS2 topic for front camera (default: /lekiwi/camera/image_raw)
        wrist_topic: ROS2 topic for wrist camera (default: /lekiwi/wrist_camera/image_raw)
        fps: Target publish rate (default: 20 Hz)
    """

    def __init__(
        self,
        sim,
        node: rclpy.node.Node,
        front_topic: str = "/lekiwi/camera/image_raw",
        wrist_topic: str = "/lekiwi/wrist_camera/image_raw",
        fps: int = 20,
    ):
        self.sim = sim
        self.node = node
        self.fps = fps
        self.dt = 1.0 / fps

        # ROS2 publishers
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.front_pub = node.create_publisher(Image, front_topic, qos)
        self.wrist_pub = node.create_publisher(Image, wrist_topic, qos)

        # cv_bridge for numpy→ROS2 Image conversion
        self.bridge = CvBridge()

        # Image cache (updated by render thread)
        self._front_img: Optional[np.ndarray] = None
        self._wrist_img: Optional[np.ndarray] = None
        self._img_lock = threading.Lock()

        # Render thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._frames_rendered = 0
        self._render_errors = 0
        self._last_render_time = 0.0

        node.get_logger().info(
            f"CameraAdapter: front={front_topic}, wrist={wrist_topic}, fps={fps}"
        )

    def start(self) -> None:
        """Start the background render thread."""
        if self._running:
            self.node.get_logger().warn("CameraAdapter: already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self.node.get_logger().info("CameraAdapter: started render thread")

    def stop(self) -> None:
        """Stop the render thread and wait for it to join."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.node.get_logger().info(
            f"CameraAdapter: stopped — rendered {self._frames_rendered} frames, "
            f"{self._render_errors} errors"
        )

    def get_stats(self) -> dict:
        """Return render statistics."""
        return {
            "frames_rendered": self._frames_rendered,
            "render_errors": self._render_errors,
            "running": self._running,
        }

    def _render_loop(self) -> None:
        """
        Background render loop — runs at self.fps Hz.
        
        Renders front + wrist cameras, updates cache, publishes ROS2 messages.
        Uses try/except to isolate MuJoCo rendering errors from the thread.
        """
        period = self.dt
        next_time = time.monotonic() + period

        while self._running:
            try:
                loop_start = time.monotonic()

                # ── Render front camera ──────────────────────────────────────────
                try:
                    front_img = self.sim.render()
                    if front_img is not None and front_img.size > 0:
                        self._publish_image(front_img, self.front_pub, "front")
                        with self._img_lock:
                            self._front_img = front_img
                except Exception as e:
                    self._render_errors += 1
                    # Don't spam logs — only log every 10 errors
                    if self._render_errors % 10 == 1:
                        self.node.get_logger().warn(
                            f"CameraAdapter: front render error #{self._render_errors}: {e}"
                        )

                # ── Render wrist camera ──────────────────────────────────────────
                try:
                    wrist_img = self.sim.render_wrist()
                    if wrist_img is not None and wrist_img.size > 0:
                        self._publish_wrist_image(wrist_img, self.wrist_pub, "wrist")
                        with self._img_lock:
                            self._wrist_img = wrist_img
                except Exception as e:
                    self._render_errors += 1
                    if self._render_errors % 10 == 1:
                        self.node.get_logger().warn(
                            f"CameraAdapter: wrist render error #{self._render_errors}: {e}"
                        )

                self._frames_rendered += 1
                self._last_render_time = time.monotonic() - loop_start

            except Exception as e:
                self._render_errors += 1
                if self._render_errors % 5 == 1:
                    self.node.get_logger().error(
                        f"CameraAdapter: render loop error #{self._render_errors}: {e}"
                    )

            # ── Sleep to maintain target fps ────────────────────────────────────
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_time += period

    def _publish_image(
        self, img: np.ndarray, pub, camera_name: str
    ) -> None:
        """Convert numpy image to ROS2 Image and publish."""
        try:
            # MuJoCo returns RGB uint8 — convert to ROS2 Image
            # cv_bridge expects BGR for Color32, but we use RGB8
            # Use sensor_msgs for direct conversion to avoid BGR dependency
            ros_msg = self._numpy_to_ros2_image(img, camera_name)
            pub.publish(ros_msg)
        except Exception as e:
            # Only log every 50 publishes to avoid spam
            if self._frames_rendered % 50 == 1:
                self.node.get_logger().warn(
                    f"CameraAdapter: publish error [{camera_name}]: {e}"
                )

    def _numpy_to_ros2_image(self, img: np.ndarray, camera_name: str) -> Image:
        """
        Convert numpy array (H, W, 3) RGB uint8 → sensor_msgs/Image.
        
        Avoids cv_bridge dependency for raw RGB images by constructing
        the ROS2 Image message directly. This is faster and avoids
        BGR/RGB confusion.
        """
        msg = Image()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = f"{camera_name}_optical_frame"
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = img.shape[1] * 3  # 3 bytes per pixel
        msg.data = img.tobytes()
        return msg

    def _publish_wrist_image(
        self, img: np.ndarray, pub, camera_name: str
    ) -> None:
        """Convert numpy image to ROS2 Image and publish (wrist variant)."""
        self._publish_image(img, pub, camera_name)

    def get_latest_front_image(self) -> Optional[np.ndarray]:
        """Get the most recent front camera frame (for VLA policy input)."""
        with self._img_lock:
            return self._front_img.copy() if self._front_img is not None else None

    def get_latest_wrist_image(self) -> Optional[np.ndarray]:
        """Get the most recent wrist camera frame."""
        with self._img_lock:
            return self._wrist_img.copy() if self._wrist_img is not None else None


# ── Standalone test ─────────────────────────────────────────────────────────────
def test_camera_adapter():
    """
    Standalone test: render 10 frames from each camera and print stats.
    Run: python3 camera_adapter.py
    """
    import sys
    sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
    
    rclpy.init(args=sys.argv)
    node = rclpy.node.Node("camera_adapter_test")

    from sim_lekiwi_urdf import LeKiWiSimURDF
    sim = LeKiWiSimURDF()
    sim.reset()

    adapter = CameraAdapter(sim, node, fps=10)
    adapter.start()

    node.get_logger().info("CameraAdapter test: rendering 2 seconds...")
    time.sleep(2.0)

    stats = adapter.get_stats()
    node.get_logger().info(f"Stats: {stats}")

    # Test get_latest_image
    front_img = adapter.get_latest_front_image()
    if front_img is not None:
        node.get_logger().info(f"Front image shape: {front_img.shape}")
    else:
        node.get_logger().warn("No front image received yet")

    adapter.stop()
    node.destroy_node()
    rclpy.shutdown()

    print(f"\nTest complete: {stats}")


if __name__ == "__main__":
    import os
    import sys
    test_camera_adapter()