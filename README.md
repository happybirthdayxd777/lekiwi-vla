# LeKiwi VLA System — Deployment Guide

Dual-mode VLA (Vision-Language-Action) robot system for LeKiwi platform, using Hugging Face LeRobot framework.

## Architecture

```
                    ┌─────────────────┐
                    │   USER INPUT    │
                    │  (Voice/Text)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   WHISPER STT   │
                    │  (Jetson Orin)  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌────────▼────────┐
     │   WORKER MODE   │          │  EXPLORER MODE  │
     │                 │          │                 │
     │ • Pi0 Policy    │          │ • Gemini Agent  │
     │ • /act endpoint  │          │ • Function Call │
     │ • Cloud GPU      │          │ • Local Logic   │
     └────────┬────────┘          └────────┬────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ HARDWARE iF     │
                    │ (LeRobot/LeKiwi)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    LEKIWI       │
                    │  (Real Robot)   │
                    └─────────────────┘
```

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **Edge (On Robot)** | Jetson Orin Nano (8GB) or similar |
| **Cloud (Server)** | GPU with 16GB+ VRAM (RTX 4000 Ada, A100) |
| **Camera** | 2x USB cameras (front + wrist) |
| **Motors** | Feetech STS3215 (6 arm + 3 wheel) |
| **Connection** | USB-C for motor bus, WiFi for cloud |

## Quick Start

### 1. Cloud Server Setup

```bash
cd server
pip install -r requirements.txt

# Download Pi0 model (~7GB)
python3 scripts/download_models.py --pi0

# Start server
./start_server.sh
```

### 2. Jetson Client Setup

```bash
# Install LeRobot
git clone https://github.com/happybirthdayxd777/lekiwi-vla.git
cd lekiwi-vla
pip install -r requirements.txt

# Configure robot connection
python3 scripts/find_port.py
python3 scripts/setup_motors.py --port /dev/ttyACM0

# Run client
python3 client/main_client.py
```

### 3. Run VLA Inference

```bash
# Test /act endpoint
curl -X POST http://localhost:8000/act \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64_encoded_image>",
    "text": "pick up the red object",
    "robot_state": {"arm_positions": [0,0,0,0,0,0], "base_velocities": [0,0,0]}
  }'
```

## Directory Structure

```
lekiwi-vla/
├── server/
│   ├── vla_server.py         # FastAPI + Pi0 inference
│   ├── requirements.txt
│   └── scripts/
│       └── download_models.py
├── client/
│   ├── main_client.py        # Orchestrator (mode switching)
│   ├── hardware_interface.py # LeRobot wrapper
│   ├── agents/
│   │   ├── worker_agent.py   # Pi0 mode
│   │   └── explorer_agent.py # Gemini mode
│   └── scripts/
├── configs/
│   └── lekiwi_vla.yaml
└── README.md
```

## Mode Switching

Press **SPACE** to toggle between modes:

| Mode | Trigger | Behavior |
|------|---------|----------|
| **WORKER** | Default | Pi0 policy, voice command execution |
| **EXPLORER** | Space | Gemini agent, free exploration |

## LeRobot Integration

This project uses [LeRobot](https://github.com/huggingface/lerobot) from Hugging Face as the underlying robotics framework.

Key LeRobot components used:
- `LeKiwi` robot class
- `Pi0Agent` for VLA inference
- `lerobot-record` for data collection
- `lerobot-teleoperate` for teleoperation

```python
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

# Connect to LeKiwi
config = LeKiwiConfig(port="/dev/ttyACM0")
robot = LeKiwi(config)
robot.connect()

# Capture observation
obs = robot.capture_observation()

# Execute action
robot.send_action(action_tensor)
```

## AI Models

| Model | Purpose | Size |
|-------|---------|------|
| Pi0 (7B) | VLA policy | 7B params |
| Pi0-ext (14B) | Extended VLA | 14B params |
| Gemini 2.0 | Explorer agent | Cloud |
| Whisper | Speech-to-text | 3B params |

## Development

```bash
# Run tests
pytest tests/

# Format code
black .

# Lint
flake8 .
```

## License

Apache 2.0