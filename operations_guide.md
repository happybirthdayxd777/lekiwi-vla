# Operations Guide: LeKiwi Hybrid Robot

## Part 1: Akamai Server (The Brain)
*Hardware: RTX 4000 Ada (20GB)*

### 1. Setup Environment
```bash
# Clone Repo
git clone <repo>
cd server

# Install (Model weights ~10GB download)
pip install -r requirements.txt
python3 scripts/download_models.py --pi0
```

### 2. Run Server
```bash
# Start FastAPI (Port 8000)
./start_server.sh
```
*Check logs ensures "Pi0 Loaded".*

## Part 2: Jetson Client (The Body)
*Hardware: Orin Nano Super (8GB)*

### 1. Hardware Prep
- **Power**: Ensure LeKiwi battery is charged.
- **Peripherals**: Plug in Realsense, Mic, Speaker, and Motor USB.
- **Network**: Connect to same WiFi/LAN as Akamai.

### 2. Install LeRobot (Official)
```bash
# 1. Install System Dependencies
sudo apt install -y portaudio19-dev libzmq3-dev
# 2. Clone LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# 3. Install LeKiwi Config
pip install -e ".[lekiwi]"
```

### 3. Configure & Calibrate (One-Time)
```bash
# 1. Find Port
lerobot-find-port
# 2. Setup Motors (Arm + Base)
lerobot-setup-motors --robot.type=lekiwi --robot.port=/dev/ttyUSB0
# 3. Calibrate
lerobot-calibrate --robot.type=lekiwi --robot.id=jetson_kiwi
```

### 4. Launch Isaac ROS (Visual Odometry)
```bash
# Terminal 1
ros2 launch isaac_ros_visual_slam ...
```

### 5. Launch Client
```bash
# Terminal 2
source venv/bin/activate
python3 client/main_client.py
```

### 4. Operation
- **Startup**: Robot enters **Worker Mode**.
  - Say "Find the bottle".
  - Pi0 receives prompt -> Drives robot towards bottle -> Picks it.
- **Interaction**: Press **Spacebar** on keyboard.
  - Robot enters **Explorer Mode**.
  - Say "Go forward".
  - Gemini triggers `move_forward()` function -> Robot moves.

## Troubleshooting
- **Jittery Movement**: Increase VSLAM confidence threshold.
- **Ignored Voice**: Check `alsamixer` on Jetson, boost Mic gain.
- **Latency**: If > 1s in Worker Mode, check Akamai network ping.
