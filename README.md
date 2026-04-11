# LeKiwi VLA System

Dual-mode VLA (Vision-Language-Action) robot system for LeKiwi platform.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WORKER MODE (Pi0)                          │
│   Voice → Whisper STT → Pi0 Policy → Action → Robot Control        │
│   "Pick up the red box" → [Image + Text] → action_delta            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        EXPLORER MODE (Gemini)                       │
│   Voice → Gemini Agent → Function Calls → Robot Control            │
│   "Go forward" → move_forward()                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Hardware Setup

- **Brain (Server)**: GPU server (RTX 4000 Ada or better) running FastAPI + Pi0
- **Body (Edge)**: Jetson Orin Nano (8GB) running client orchestrator

## Quick Start

### Server Setup
```bash
cd server
pip install -r requirements.txt
python3 scripts/download_models.py --pi0
./start_server.sh
```

### Jetson Client Setup
```bash
# Install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[lekiwi]"

# Configure
lerobot-setup-motors --robot.type=lekiwi --robot.port=/dev/ttyACM0

# Launch
python3 client/main_client.py
```

## Robot Overview

LeKiwi has:
- **3 omni wheels** (motor IDs 7, 8, 9) - differential drive mobile base
- **6-DOF arm** (motor IDs 1-6) - STS3215 servos
- **2 cameras**: front + wrist

## Mode Switching

Press **SPACE** to toggle between Worker and Explorer modes.

## License

Apache 2.0