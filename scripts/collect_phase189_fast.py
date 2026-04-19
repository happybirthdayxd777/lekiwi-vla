#!/usr/bin/env python3
"""
Phase 189 FAST data collection — Per-step images at small resolution.
FIXES Phase 187 bug: now saves ONE image per timestep (not 1 per episode).

Usage:
  python3 scripts/collect_phase189_fast.py --episodes 50 --output data/phase189_clean_50ep.h5
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
import mujoco
from datetime import datetime

RENDER_W, RENDER_H = 320, 240

# ── MuJoCo model ──────────────────────────────────────────────────────────────
_MINIMAL_XML = """
<mujoco model="lekiwi">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="Euler" jacobian="dense"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1" castshadow="false"/>
    <geom type="plane" size="5 5 0.01" rgba=".9 .9 .9 1" friction="1 0.005 0.001" condim="1"/>
    <body name="base" pos="0 0 0.15">
      <freejoint/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.001 0.001 0.001"/>
      <geom type="box" size="0.08 0.10 0.05" rgba="0.1 0.2 0.8 1" friction="2 0.005 0.001" condim="3"/>
      <body name="w1" pos="-0.087 -0.100 0" euler="0 0 0">
        <joint name="w1_j" type="hinge" axis="1 0 0" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-6 1e-6 1e-6"/>
        <geom type="cylinder" size="0.042 0.016" rgba="0.05 0.05 0.05 1" friction="2 0.005 0.001" condim="3"/>
      </body>
      <body name="w2" pos=" 0.087 -0.100 0" euler="0 0 0">
        <joint name="w2_j" type="hinge" axis="1 0 0" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-6 1e-6 1e-6"/>
        <geom type="cylinder" size="0.042 0.016" rgba="0.05 0.05 0.05 1" friction="2 0.005 0.001" condim="3"/>
      </body>
      <body name="w3" pos="0 0.100 0" euler="0 0 0">
        <joint name="w3_j" type="hinge" axis="0 0 1" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-6 1e-6 1e-6"/>
        <geom type="cylinder" size="0.042 0.016" rgba="0.05 0.05 0.05 1" friction="2 0.005 0.001" condim="3"/>
      </body>
      <body name="arm0" pos="0 -0.10 0.06">
        <joint name="arm0_j" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="1e-5 1e-5 1e-5"/>
        <geom type="capsule" size="0.015 0.06" rgba="0.6 0.3 0.1 1"/>
        <body name="arm1" pos="0 0.06 0">
          <joint name="arm1_j" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.5"/>
          <inertial pos="0 0 0" mass="0.15" diaginertia="1e-5 1e-5 1e-5"/>
          <geom type="capsule" size="0.012 0.05" rgba="0.6 0.3 0.1 1"/>
          <body name="arm2" pos="0 0.05 0">
            <joint name="arm2_j" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.5"/>
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-5 1e-5 1e-5"/>
            <geom type="capsule" size="0.010 0.04" rgba="0.6 0.3 0.1 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="w1_j" gear="1" ctrllimited="true" ctrlrange="-0.5 0.5"/>
    <motor joint="w2_j" gear="1" ctrllimited="true" ctrlrange="-0.5 0.5"/>
    <motor joint="w3_j" gear="1" ctrllimited="true" ctrlrange="-0.5 0.5"/>
    <motor joint="arm0_j" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="arm1_j" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="arm2_j" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <worldbody>
    <camera name="front" pos="0 -0.5 0.5" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""

def twist_to_contact_wheel_speeds(vx, vy, wz=0.0):
    """Phase 164 calibrated IK for k_omni=15.0 overlay."""
    vx_200 = vx * 200.0
    vy_200 = vy * 200.0
    w1 = -0.0124 * vx_200 + 0.1880 * vy_200
    w2 =  0.1991 * vx_200 + 0.1991 * vy_200
    w3 = -0.1993 * vx_200 + 0.1872 * vy_200
    return np.clip(np.array([w1, w2, w3]), -0.5, 0.5)

class CleanJacobianController:
    def __init__(self, kP=0.5, wheel_clip=0.5):
        self.kP = kP
        self.wheel_clip = wheel_clip

    def compute_wheel_velocities(self, base_xy, goal_xy):
        dx = goal_xy[0] - base_xy[0]
        dy = goal_xy[1] - base_xy[1]
        dist = np.linalg.norm([dx, dy])
        if dist < 0.005:
            return np.zeros(3, dtype=np.float32)
        vx = self.kP * dx
        vy = self.kP * dy
        wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
        if self.wheel_clip is not None:
            wheel_speeds = np.clip(wheel_speeds, -self.wheel_clip, self.wheel_clip)
        return np.array(wheel_speeds, dtype=np.float32)

class FastSim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_MINIMAL_XML)
        self.data = mujoco.MjData(self.model)
        self._base_id = self.model.body('base').id
        self._w1_id = self.model.joint('w1_j').id
        self._w2_id = self.model.joint('w2_j').id
        self._w3_id = self.model.joint('w3_j').id
        self._arm_ids = [self.model.joint(f'arm{i}_j').id for i in range(3)]
        self._renderer = mujoco.Renderer(self.model, RENDER_W, RENDER_H)
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:13] = 0
        return self._obs()

    def _obs(self):
        arm_q = self.data.qpos[7:13].copy()
        wheel_v = np.array([
            self.data.qvel[self._w1_id],
            self.data.qvel[self._w2_id],
            self.data.qvel[self._w3_id],
        ])
        return {'arm_positions': arm_q, 'wheel_velocities': wheel_v}

    def step(self, action):
        # action: 9D [arm0..2, w1, w2, w3, 0, 0, 0]
        ctrl = np.zeros(9)
        ctrl[:3] = action[:3]  # arm
        ctrl[3:6] = action[3:6]  # wheel (actual ctrl indices for w1,w2,w3)
        # But actuator order: w1,w2,w3,arm0,arm1,arm2 = indices 0..5
        # Let me check actuator order
        ctrl_6 = np.zeros(6)
        ctrl_6[0:3] = action[3:6]  # w1,w2,w3
        ctrl_6[3:6] = action[:3]   # arm0,arm1,arm2
        self.data.ctrl[:] = ctrl_6
        mujoco.mj_step(self.model, self.data)
        dist = np.linalg.norm(self.data.qpos[:2])
        reward = 1.0 if dist < 0.1 else 0.0
        return self._obs(), reward, dist < 0.05, {}

    def render(self):
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render()

    def close(self):
        self._renderer.close()

def collect_episode_clean(sim, controller, goal_pos, max_steps=200, arm_action_scale=0.05):
    obs = sim.reset()
    for _ in range(15):
        sim.step(np.zeros(9))

    goal_norm = np.clip(goal_pos / 0.5, -1, 1).astype(np.float32)
    arm_pos = np.zeros(6, dtype=np.float32)

    states_list, actions_list, rewards_list, goals_list, images_list = [], [], [], [], []

    for step in range(max_steps):
        base_xy = sim.data.xpos[sim._base_id, :2]
        wheel_vels = controller.compute_wheel_velocities(base_xy, goal_pos)

        # Arm random walk
        arm_delta = np.random.normal(0, arm_action_scale, size=6).astype(np.float32)
        arm_pos = np.clip(arm_pos + arm_delta, -1.0, 1.0)

        # Build 9D action [arm0..5, w1, w2, w3]
        arm_action = arm_pos.astype(np.float32)
        wheel_action = (wheel_vels / 0.5).astype(np.float32)
        wheel_action = np.clip(wheel_action, -1.0, 1.0)
        action = np.concatenate([arm_action, wheel_action])

        obs, reward, done, _ = sim.step(action)

        # Build 11D state: arm_pos(6) + wheel_vel(3) + goal_norm(2)
        wheel_v_raw = obs['wheel_velocities']
        wheel_v_norm = np.clip(wheel_v_raw / 0.5, -1, 1).astype(np.float32)
        arm_pos_norm = np.clip(arm_pos / 2.0, -1, 1).astype(np.float32)
        state11 = np.concatenate([arm_pos_norm, wheel_v_norm, goal_norm])

        img = sim.render()

        states_list.append(state11)
        actions_list.append(action)
        rewards_list.append(reward)
        goals_list.append(goal_norm)
        images_list.append(img)

        if done:
            break

    return {
        'states': np.array(states_list, dtype=np.float32),
        'actions': np.array(actions_list, dtype=np.float32),
        'rewards': np.array(rewards_list, dtype=np.float32),
        'goal_norm': np.array(goals_list, dtype=np.float32),
        'goal_world': np.array([goal_pos] * len(states_list), dtype=np.float32),
        'images': images_list,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', type=str, default='data/phase189_clean_50ep.h5')
    parser.add_argument('--goal_min', type=float, default=0.2)
    parser.add_argument('--goal_max', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    controller = CleanJacobianController(kP=0.5, wheel_clip=0.5)
    sim = FastSim()

    all_states, all_actions, all_rewards, all_goals_norm = [], [], [], []
    all_goals_world, all_images = [], []
    episode_starts = [0]
    total_frames = 0
    successes = 0

    print(f"Phase 189 FAST: {args.episodes}ep x {args.steps}steps ({RENDER_W}x{RENDER_H} images)...")

    for ep in range(args.episodes):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(args.goal_min, args.goal_max)
        goal_pos = np.array([r*np.cos(angle), r*np.sin(angle)], dtype=np.float32)

        ep_data = collect_episode_clean(sim, controller, goal_pos, max_steps=args.steps,
                                        arm_action_scale=0.05)

        n = len(ep_data['states'])
        all_states.append(ep_data['states'])
        all_actions.append(ep_data['actions'])
        all_rewards.append(ep_data['rewards'])
        all_goals_norm.append(ep_data['goal_norm'])
        all_goals_world.append(ep_data['goal_world'])
        all_images.extend(ep_data['images'])  # FIXED: per-step images

        episode_starts.append(total_frames + n)
        total_frames += n

        if ep_data['rewards'].sum() > 0:
            successes += 1

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{args.episodes}: {n} frames, cum={total_frames}, successes={successes}")

    print(f"\nTotal frames: {total_frames}, SR: {successes}/{args.episodes}")

    all_states = np.concatenate(all_states, axis=0).astype(np.float32)
    all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    all_rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
    all_goals_norm = np.concatenate(all_goals_norm, axis=0).astype(np.float32)
    all_goals_world = np.concatenate(all_goals_world, axis=0).astype(np.float32)
    all_images_stacked = np.stack(all_images, axis=0).astype(np.uint8)

    # Quality check
    print("\n=== DATA QUALITY CHECK ===")
    w = all_actions[:, 6:9]
    g = all_goals_world
    for i, n in enumerate(['w0', 'w1', 'w2']):
        cx = np.corrcoef(w[:, i], g[:, 0])[0, 1]
        cy = np.corrcoef(w[:, i], g[:, 1])[0, 1]
        print(f"  {n}: corr_x={cx:+.3f} corr_y={cy:+.3f}")

    print(f"\n  Images: {all_images_stacked.shape} — should equal {total_frames} frames")
    assert all_images_stacked.shape[0] == total_frames, f"MISMATCH: {all_images_stacked.shape[0]} images vs {total_frames} states!"

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print(f"\nSaving to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('states', data=all_states, compression='gzip')
        f.create_dataset('actions', data=all_actions, compression='gzip')
        f.create_dataset('rewards', data=all_rewards, compression='gzip')
        f.create_dataset('goal_positions', data=all_goals_world, compression='gzip')
        f.create_dataset('goal_norm', data=all_goals_norm, compression='gzip')
        f.create_dataset('images', data=all_images_stacked, compression='gzip')
        f.create_dataset('episode_starts', data=np.array(episode_starts))
        f.attrs['phase'] = 189
        f.attrs['controller'] = 'CleanJacobianController (kP=0.5)'
        f.attrs['state_format'] = 'arm_pos(6)+wheel_vel(3)+goal_norm(2)=11D'
        f.attrs['action_format'] = 'arm(6)+wheel(3)=9D'
        f.attrs['render'] = f'{RENDER_W}x{RENDER_H}'
        f.attrs['created'] = datetime.now().isoformat()

    print(f"\nDataset: states={all_states.shape}, images={all_images_stacked.shape}")
    print(f"Rewards: {int((all_rewards>0).sum())}/{len(all_rewards)} positive")
    print(f"\n✓ Saved: {args.output}")
    sim.close()

if __name__ == '__main__':
    main()
