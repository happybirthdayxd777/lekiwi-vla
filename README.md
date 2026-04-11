# LeKiwi VLA Project

Autonomous robot learning research platform combining MuJoCo simulation, LeRobot VLA training/inference, and real robot control.

## Architecture Overview

```
lekiwi_vla/
├── sim_lekiwi.py              # MuJoCo simulation + Gymnasium wrapper
├── lerobot_policy_inference.py # LeRobot VLA inference (ACT/Diffusion/Flow Matching)
├── policies/
│   └── lerobot_flow_matching.yaml  # Multi-Task DiT training config
├── scripts/
│   ├── record_lekiwi.py       # Data recording with LeRobot + SO-101 leader
│   ├── train_flow_matching.py # Train Flow Matching policy
│   └── eval_policy.py         # Evaluate any LeRobot policy on sim or real
├── docs/
│   └── VLA_COMPARISON.md      # Full VLA architecture comparison report
└── results/                   # Training checkpoints
```

## Quick Start

```bash
# Simulate with mock policy
python3 lerobot_policy_inference.py --policy mock --steps 100

# Record teleoperation data
python3 scripts/record_lekiwi.py \
  --hf-repo-id <your_name>/lekiwi-demo \
  --task "walk to the red marker" \
  --episodes 10

# Train Flow Matching policy (4-step inference!)
python3 scripts/train_flow_matching.py \
  --dataset <your_name>/lekiwi-demo \
  --output results/lekiwi_fm \
  --epochs 100

# Evaluate on simulation
python3 scripts/eval_policy.py \
  --policy multi_task_dit \
  --checkpoint results/lekiwi_fm/checkpoints/latest \
  --dataset <your_name>/lekiwi-demo \
  --sim
```

## Supported Policies

| Policy | Action Generation | Inference Steps | LeRobot | Notes |
|--------|------------------|-----------------|---------|-------|
| **ACT** | Action Chunking | 1 | ✅ | Fast, good baseline |
| **Diffusion** | DDPM | 50-100 | ✅ | Smooth but slow |
| **Multi-Task DiT** | Flow Matching | **4** | ✅ | Best speed/quality |
| **GR00T-N1.5** | Flow Matching + DiT | 4 | ✅ | nvidia/GR00T-N1.5-3B |
| **SmolVLA** | Regression | 1 | ✅ | ~1B params, edge device |
| **Pi0** | Diffusion | 10+ | ✅ | Closed, most capable |

See `docs/VLA_COMPARISON.md` for full analysis.

## Simulated Robot: LeKiwi

- **9 DOF**: 6 arm joints + 3 omni wheels
- **Gymnasium compatible**: `LeKiwiEnv(env_id="lekiwi/...)`
- **Observation**: camera image + arm positions + wheel velocities
- **Action**: 6 joint targets + 3 wheel velocities

```python
from sim_lekiwi import LeKiwiSim

sim = LeKiwiSim()
sim.reset()

for step in range(200):
    img = sim.render()
    obs = sim._obs()
    action = policy.predict(img, obs)
    obs = sim.step(action)
    reward = sim.get_reward()
```

## Research Findings

See `docs/VLA_COMPARISON.md` for:
- Deep dive: Flow Matching vs Diffusion vs ACT
- UnifoLM-VLA architecture analysis (cloned to `~/unifolm-vla/`)
- Unitree G1 dataset evaluation
- GR00T-N1.5 vs Pi0 vs OpenVLA benchmark
- Hardware recommendations for your research

## Related Projects

| Project | Path | Purpose |
|---------|------|---------|
| UnifoLM-VLA research | `~/unifolm-vla/` | Unitree's Flow Matching VLA (cloned) |
| Go2 VLA | `~/go2-vla/` | Quadruped gait controller + ROS2 |
| Robot CTF Workshop | `~/robot-security-workshop/` | Security research + Docker |
| Adversarial Toolkit | `~/robot-security-workshop/adversarial_toolkit/` | Robot attack toolkit |
| LeRobot | `~/lerobot/` | HF LeRobot library (local install) |