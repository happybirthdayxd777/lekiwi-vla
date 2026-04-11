# LeKiwi VLA — Visual Learning for Autonomous Robots

Research platform for training and deploying Vision-Language-Action (VLA) models on the LeKiwi robot platform using LeRobot.

## Project Structure

```
lekiwi_vla/
├── sim_lekiwi.py               # MuJoCo simulation + Gymnasium wrapper
├── lerobot_policy_inference.py # LeRobot VLA inference (ACT/Diffusion/Flow Matching/GR00T)
├── scripts/
│   ├── record_lekiwi.py           # Data recording with LeRobot + SO-101 leader arm
│   ├── train_flow_matching.py      # LeRobot Multi-Task DiT training
│   ├── train_flow_matching_lekiwi.py  # Standalone Flow Matching (no LeRobot deps)
│   └── eval_policy.py              # Evaluate any policy on sim or real
├── policies/
│   └── lerobot_flow_matching.yaml  # Multi-Task DiT (Flow Matching) config
└── docs/
    └── VLA_COMPARISON.md          # Full VLA architecture comparison
```

## Quick Start

```bash
# Simulate with mock policy (no hardware needed)
python3 lerobot_policy_inference.py --policy mock --steps 100

# Record teleoperation data
python3 scripts/record_lekiwi.py \
  --hf-repo-id <your_name>/lekiwi-demo \
  --task "walk to the red marker and stop" \
  --episodes 10

# Train Flow Matching policy (4-step inference!)
python3 scripts/train_flow_matching_lekiwi.py \
  --dataset <hf_repo> \
  --output ../results/lekiwi_fm \
  --epochs 100

# Evaluate on simulation
python3 scripts/eval_policy.py \
  --policy multi_task_dit \
  --checkpoint ../results/lekiwi_fm/checkpoints/latest \
  --dataset <hf_repo> \
  --sim
```

## Supported VLA Policies

| Policy | Method | Inference | LeRobot | Notes |
|--------|--------|-----------|---------|-------|
| **ACT** | Action Chunking | 1 step | ✅ | Fast baseline |
| **Diffusion** | DDPM | 50-100 steps | ✅ | Smooth but slow |
| **Multi-Task DiT** | Flow Matching | **4 steps** | ✅ | Best speed/quality |
| **GR00T-N1.5** | Flow Matching + DiT | 4 steps | ✅ | nvidia/GR00T-N1.5-3B |
| **SmolVLA** | Regression | 1 step | ✅ | ~1B params, edge |
| **Pi0** | Diffusion | 10+ steps | ✅ | Closed, most capable |

See `docs/VLA_COMPARISON.md` for full analysis.

## LeKiwi Simulation (MuJoCo)

- **9 DOF**: 6 arm joints + 3 omni wheels
- **Gymnasium compatible**: `LeKiwiEnv(env_id="hermes_research/LeKiwi-v0")`
- **Observation**: camera image (224×224) + arm positions (6) + wheel velocities (3)
- **Action**: 6 arm joint targets + 3 wheel velocities

```python
import gymnasium as gym
from sim_lekiwi import LeKiwiSim

gym.register("hermes_research/LeKiwi-v0", LeKiwiSim, max_episode_steps=200)
env = gym.make("hermes_research/LeKiwi-v0")
obs, _ = env.reset()

img = obs["image"]    # PIL Image
state = obs["state"]  # [9] numpy array

action = your_policy(img, state)
obs, reward, term, trunc, info = env.step(action)
```

## Hardware Platform (lekiwi_modular)

Real robot ROS2 code is in `../lekiwi_modular/`:
- Full URDF + 3D meshes (STL)
- ROS2 controllers (omni_controller, odometry)
- Docking experiment data (55+ runs)
- IMU data with real noise characteristics

See `../lekiwi_modular/RESEARCH_ANALYSIS.md` for full details.

## Research Findings

### Flow Matching (4-Step Inference)
Flow Matching enables 4-step inference (vs 100 steps for DDPM):
- Training: predict velocity = (x_clean - x_noise)
- Inference: Euler ODE integration in 4 steps
- Key insight: `repeated_diffusion_steps=8` improves sample efficiency

### GR00T-N1.5 (NVIDIA)
- Available on HuggingFace: `nvidia/GR00T-N1.5-3B`
- Architecture: Flow Matching + DiT + Qwen2.5-VL
- 3B parameters, ~8GB VRAM, 4-step inference
- LeRobot integration: `GrootPolicy`

### Key Bug Found in omni_controller.py
The original `omni_controller.py` used identical `joint_axes` for all three wheels.
See `../lekiwi_modular/src/lekiwi_controller/scripts/omni_controller_fixed.py` for corrected version.

## Related Projects

| Project | Location | Purpose |
|---------|----------|---------|
| lekiwi_modular | `../lekiwi_modular/` | Real robot ROS2 code + URDF |
| go2-vla | `../go2-vla/` | Unitree Go2 quadruped gait + ROS2 |
| robot-security-workshop | `../robot-security-workshop/` | Robot security CTF + adversarial toolkit |
| unifolm-vla | `../unifolm-vla/` | Unitree VLA (Flow Matching + DiT) |
| unitree_rl_gym | `../unitree_rl_gym/` | Isaac Gym RL training for Go2/G1 |
| lerobot | `~/lerobot/` | HuggingFace LeRobot library (local install) |