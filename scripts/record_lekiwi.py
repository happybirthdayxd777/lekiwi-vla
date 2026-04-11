#!/usr/bin/env python3
"""
LeKiwi Data Recording Script
===========================
Records teleoperation data using LeRobot library.

Hardware required:
  - LeKiwi robot (with LeKiwiHost running on the robot)
  - SO-101 leader arm connected via USB
  - Keyboard for emergency stop / episode control

Usage:
  # Record 10 episodes of "pick and place"
  python3 scripts/record_lekiwi.py \
    --hf-repo-id <your_username>/lekiwi-demo \
    --task "pick the red block and place in the blue bowl" \
    --episodes 10 \
    --episode-time 30

  # Push to HuggingFace after recording
  python3 scripts/record_lekiwi.py --push ...
"""

import argparse
import sys
from pathlib import Path

# Add LeRobot to path
sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10


def main():
    parser = argparse.ArgumentParser(description="Record LeKiwi teleoperation data")
    parser.add_argument("--hf-repo-id", type=str, required=True,
                        help="HuggingFace repo ID, e.g. your_name/lekiwi-demo")
    parser.add_argument("--task", type=str, required=True,
                        help="Task description in natural language")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to record")
    parser.add_argument("--episode-time", type=int, default=EPISODE_TIME_SEC,
                        help="Episode duration in seconds")
    parser.add_argument("--reset-time", type=int, default=RESET_TIME_SEC,
                        help="Reset time between episodes")
    parser.add_argument("--robot-ip", type=str, default="172.18.134.136",
                        help="LeKiwi robot IP address")
    parser.add_argument("--arm-port", type=str, default="/dev/tty.usbmodem585A0077581",
                        help="SO-101 arm serial port")
    parser.add_argument("--push", action="store_true",
                        help="Push dataset to HuggingFace after recording")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test without connecting to hardware")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  LeKiwi Data Recording")
    print(f"  Task    : {args.task}")
    print(f"  Episodes: {args.episodes}")
    print(f"  FPS     : {FPS}")
    print(f"  Device  : {'DRY RUN (mock)' if args.dry_run else 'REAL HARDWARE'}")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN] Would record with:")
        print(f"  Robot IP : {args.robot_ip}")
        print(f"  Arm Port : {args.arm_port}")
        print(f"  Repo ID  : {args.hf_repo_id}")
        return

    # ── Robot & Teleoperator Config ──────────────────────────────────────
    robot_config = LeKiwiClientConfig(remote_ip=args.robot_ip, id="lekiwi")
    leader_arm_config = SO100LeaderConfig(port=args.arm_port, id="leader_arm")
    keyboard_config = KeyboardTeleopConfig()

    # ── Initialize Devices ─────────────────────────────────────────────────
    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # ── Dataset Setup ────────────────────────────────────────────────────
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=args.hf_repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # ── Connect & Record ─────────────────────────────────────────────────
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    try:
        if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print(f"\nRecording {args.episodes} episodes of: '{args.task}'")
        print("Press SPACE to pause, Q to quit\n")

        recorded = 0
        while recorded < args.episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded + 1}")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=args.episode_time,
                single_task=args.task,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if not events["stop_recording"] and (
                (recorded < args.episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=[leader_arm, keyboard],
                    control_time_s=args.reset_time,
                    single_task=args.task,
                    display_data=True,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded += 1

    finally:
        log_say("Stop recording")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()
        dataset.finalize()

        if args.push:
            print("\n[Pushing dataset to HuggingFace...]")
            dataset.push_to_hub()
            print("Done!")


if __name__ == "__main__":
    main()