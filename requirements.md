# Requirements: LeKiwi Hybrid VLA Robot

> **Reference**: 
> - [Official LeKiwi Installation Guide](https://huggingface.co/docs/lerobot/en/lekiwi)
> - [LeKiwi Orchestration Example](https://github.com/tinjyuu/lerobot/blob/321a587ec74213705d9bbe184fde88adfc35d90e/examples/lekiwi/orchestrate.py)
> - [xLeRobot LLM Agent Control](https://xlerobot.readthedocs.io/en/latest/software/getting_started/LLM_agent.html)

## 1. System Overview
A **Dual-Mode Generalist Robot** built on the **LeKiwi (SO-101)** platform.
- **Goal**: Combine precise manifestation (Pi0) with open-ended exploration (Gemini).
- **Core Philosophy**: "Local Hearing, Cloud Thinking, Hybrid Action."

## 2. Hardware Specification
### Robot Platform
- **Model**: **LeKiwi (SO-101)** 6-DOF Arm + Mobile Base.
- **Sensors**:
  - **Wrist Camera**: RealSense D435/USB (640x480, 30fps - Input for π0).
  - **Head Camera**: Wide-angle RGB (1280x720, 30fps - Input for Gemini + VSLAM).
  - **Microphone & Speaker**: USB Interface (Local interaction, 16kHz sampling).
  - **IMU**: 9-axis (for odometry and stability).
- **Compute (Edge)**: **Jetson Orin Nano Super (8GB)**.
  - **CPU**: 6-core ARM Cortex-A78AE
  - **GPU**: 1024-core NVIDIA Ampere with 32 Tensor Cores
  - **Memory**: 8GB LPDDR5
  - **Power**: 15W-25W (configurable)

### Cloud Infrastructure (The Brain)
- **Primary Server**: **Akamai GPU Instance** (Cloud Compute).
  - **GPU**: NVIDIA RTX 4000 Ada Generation (20GB VRAM, 6144 CUDA cores).
  - **CPU**: 16 vCPUs
  - **RAM**: 64GB
  - **Network**: 10Gbps uplink
  - **Hosting**:
    - **Action Model**: **OpenPI / π0 (Pi-Zero)** (7B VLA, bfloat16).
      - **Architecture**: Vision-Language-Action Transformer
      - **Training**: Physical Intelligence's robot manipulation dataset
      - **Input**: RGB images (224x224) + text instruction
      - **Output**: 7-DOF action vectors at 50Hz (position, rotation, gripper)
    - *Note*: π0 is a unified VLA model; no separate vision encoder needed.
- **Secondary API**: **Google Gemini 1.5 Pro** (for "Explorer Mode" fallback and autonomous behavior).
  - **Context Window**: 1M tokens (supports long video/audio streams)
  - **Multimodal**: Native video, audio, and text understanding

## 3. Operational Modes

### Mode 1: "The Worker" (OpenPI/π0 VLA)
*Direct perception-to-action with vision-language grounding.*
- **Trigger**: Default State or explicit manipulation commands.
- **Model**: **OpenPI (π0)** - 7B parameter Vision-Language-Action transformer
- **Loop**:
  1.  **Hear**: Local **Whisper** (Jetson Orin Nano) transcribes voice command
      - Example: "Pick up the red cup and place it on the table"
      - Model: `faster-whisper-base` running on Jetson GPU
      - Latency: <500ms for typical commands
  2.  **See**: Capture visual context
      - **Wrist Camera**: Task-focused view (manipulation target)
      - **Head Camera**: Environmental context (workspace overview)
      - Images resized to 224x224 for π0 input
  3.  **Think**: Cloud π0 inference on Akamai GPU
      - **Input**: RGB image tensor + tokenized text instruction
      - **Processing**: Vision-language fusion → action policy
      - **Output**: Action sequence (joint positions, velocities, gripper state)
      - **Frequency**: 50Hz action chunks
  4.  **Act**: Jetson executes motor commands
      - **Arm Control**: 6-DOF joint trajectories
      - **Base Control**: Differential drive commands (if needed)
      - **Gripper**: Binary open/close or continuous force control
- **Strengths**:
  - High precision for learned manipulation tasks
  - Visual grounding (can distinguish "red cup" from "blue cup")
  - Robust to lighting and viewpoint changes
- **Limitations**:
  - Requires visual line-of-sight to target
  - Limited to task distribution seen in training
  - Long-range navigation not in π0 training data
  - **Fallback Condition**: If π0 confidence <0.3 or task requires navigation → switch to Explorer Mode

### Mode 2: "The Explorer" (Gemini LLM Agent)
*Interactive, curious, and self-driven - Inspired by xLeRobot LLM Agent Control.*
- **Trigger**:
  - Manual toggle (Keyboard 'Space' or API)
  - Automatic fallback when π0 cannot complete task
  - Explicit navigation or exploration commands
- **Model**: **Google Gemini 1.5 Pro** (via API)
- **Architecture Pattern**: Based on xLeRobot LLM Agent framework
  - **Reference**: [xLeRobot LLM Agent Control](https://xlerobot.readthedocs.io/en/latest/software/getting_started/LLM_agent.html)
  - **Principle**: LLM acts as high-level planner, calls low-level motor primitives
- **Loop**:
  1.  **Sense**: Multimodal streaming to Gemini
      - **Video**: Head camera stream (5fps, JPEG compressed)
      - **Audio**: Microphone input (continuous or voice-activated)
      - **State**: Robot pose, battery level, task history (via system prompt)
  2.  **Think**: Gemini processes context and generates plan
      - **System Prompt**: Defines robot persona, capabilities, and safety rules
      - **Example Prompt**:
        ```
        You are Lekiwi, a helpful robot assistant. You can see through your camera
        and move using primitive functions. You are curious and friendly.
        Safety rules: Never move faster than 0.2m/s. Stop if you detect obstacles.
        Available functions: move_forward(distance), turn_left(angle), turn_right(angle),
        move_backward(distance), speak(text), pick_object(description), place_object(location).
        ```
      - **Output**: Natural language response + function calls
  3.  **Respond**:
      - **Voice**: Text-to-speech via Piper TTS on Jetson
        - Example: "I see a doorway ahead. I'll navigate through it."
      - **Motor Commands**: **Function Calling Interface** (xLeRobot pattern)
        - Gemini uses native function calling to invoke motor primitives
        - **Available Functions**:
          - `move_forward(distance_m: float)` - Move forward by distance
          - `move_backward(distance_m: float)` - Move backward by distance
          - `turn_left(angle_deg: float)` - Rotate counterclockwise
          - `turn_right(angle_deg: float)` - Rotate clockwise
          - `stop()` - Emergency stop
          - `pick_object(description: str)` - Delegate to π0 VLA mode
          - `place_object(location: str)` - Delegate to π0 VLA mode
          - `speak(text: str)` - Verbal response
          - `get_camera_view()` - Request current camera frame
        - **Execution**: Jetson receives function calls and executes motion primitives
        - **Feedback Loop**: Function results returned to Gemini for next decision
  4.  **Monitor**: Safety and state tracking
      - **Obstacle Detection**: RealSense depth camera → emergency stop if <0.5m
      - **Speed Limiter**: All motions capped at 0.2m/s (explorer mode)
      - **Timeout**: If Gemini doesn't respond in 10s, return to idle
      - **Cost Control**: Limit API calls to 100 requests/hour
- **Strengths**:
  - Open-ended task understanding and planning
  - Natural human interaction and explanation
  - Can handle novel situations not in π0 training
  - Long-horizon planning (multi-step navigation)
- **Limitations**:
  - Higher latency than π0 (1-3 seconds per decision)
  - Potential for hallucinated actions (mitigated by safety limits)
  - Requires internet connectivity
  - API costs scale with usage
- **Fallback Strategy**:
  - If Gemini API unavailable → enter safe mode (stop and alert user)
  - For manipulation subtasks → delegate back to π0 Worker Mode
  - Example: "Navigate to the kitchen [Gemini] and pick up the red mug [π0]"

## 4. Software Stack

### Jetson Orin Nano (Edge Client)
- **Robot Control Framework**:
  - `lerobot[lekiwi]` (Official LeRobot SDK for SO-101 hardware)
  - `ros2 humble` (Robot Operating System 2)
  - `isaac_ros_visual_slam` (NVIDIA visual odometry and mapping)
- **Speech & Audio**:
  - `faster-whisper` (Base or Small model, GPU-accelerated STT)
  - `piper-tts` (Local text-to-speech, low-latency)
  - `sounddevice` (USB audio I/O)
- **Vision Processing**:
  - `pyrealsense2` (RealSense camera SDK)
  - `opencv-python` (Image preprocessing)
  - `librealsense` (Depth and RGB stream handling)
- **Communication**:
  - `websockets` (Real-time streaming to Akamai server)
  - `google-genai` (Gemini API client)
  - `requests` (HTTP client for VLA server)
- **Orchestration**:
  - `orchestrate.py` (Main control loop, mode switching logic)
  - Based on: [LeKiwi Orchestration Example](https://github.com/tinjyuu/lerobot/blob/321a587ec74213705d9bbe184fde88adfc35d90e/examples/lekiwi/orchestrate.py)

### Akamai Cloud GPU Server
- **VLA Inference Server**:
  - `vla_server.py`: FastAPI application serving π0 model
  - `torch` (PyTorch 2.1+, CUDA 12.1)
  - `transformers` (HuggingFace, for π0 model loading)
  - `pi_zero_pytorch` (OpenPI model weights and inference code)
  - **Endpoints**:
    - `POST /predict`: Accepts image + text → returns action vector
    - `GET /health`: Health check and model status
    - `POST /reset`: Reset internal state (for multi-step tasks)
- **Authentication & Security**:
  - HTTPS with Let's Encrypt SSL certificate
  - Bearer token authentication
  - Rate limiting: 100 req/min per client
- **Deployment**:
  - Docker container with NVIDIA runtime
  - Auto-scaling based on GPU utilization (future)

### Cloud APIs
- **Google Gemini 1.5 Pro**:
  - SDK: `google-generativeai` Python library
  - API Key management via environment variables
  - Function calling schema defined in `gemini_tools.json`

## 5. Mode Switching & Orchestration

### Switching Logic
- **Manual Toggle**: Keyboard input (`Space` key) or HTTP API call
- **Automatic Fallback**:
  ```python
  if mode == "worker" and (pi0_confidence < 0.3 or "navigate" in command):
      switch_to_explorer_mode()
  elif mode == "explorer" and ("pick" in command or "grasp" in command):
      switch_to_worker_mode()
  ```
- **Priority**: User manual override > automatic fallback > default (worker mode)

### Hybrid Task Execution
Example: "Go to the kitchen and pick up the red mug"
1. **Gemini Explorer Mode**: Parse command → recognize navigation + manipulation
2. **Gemini**: Execute navigation to kitchen using `move_forward()`, `turn_left()`
3. **Auto-Switch**: Once in kitchen, delegate manipulation to π0
4. **π0 Worker Mode**: Visual grounding on "red mug" → pick and grasp
5. **Completion**: Return control to Gemini or await next command

### Status Indicators
- **Terminal Output**:
  - `[LISTENING]` - Waiting for voice command (Whisper active)
  - `[THINKING - PI0]` - Cloud VLA inference in progress
  - `[THINKING - GEMINI]` - LLM reasoning in progress
  - `[ACTING]` - Executing motor commands
  - `[IDLE]` - Standby mode
- **LED Indicators** (Optional hardware mod):
  - Blue: Listening
  - Yellow: Thinking
  - Green: Acting
  - Red: Error/Safety stop

## 6. System Integration

### Data Flow Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                            │
│                    (Voice/Keyboard)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              JETSON ORIN NANO (Edge)                         │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │   Whisper    │  │  Cameras    │  │ Motor Ctrl   │       │
│  │     STT      │  │ (RGB+Depth) │  │   (ROS2)     │       │
│  └──────┬───────┘  └──────┬──────┘  └──────▲───────┘       │
│         │                 │                 │                │
│         └────────►┌───────┴─────────┐◄──────┘                │
│                   │  orchestrate.py  │                       │
│                   │ (Mode Switcher)  │                       │
│                   └────┬────────┬────┘                       │
└────────────────────────┼────────┼──────────────────────────┘
                         │        │
         ┌───────────────┘        └──────────────┐
         │                                        │
         ▼                                        ▼
┌────────────────────┐                 ┌────────────────────┐
│  AKAMAI GPU SERVER │                 │   GEMINI API       │
│  ┌──────────────┐  │                 │  ┌──────────────┐  │
│  │   π0 VLA     │  │                 │  │ Gemini 1.5   │  │
│  │   Model      │  │                 │  │     Pro      │  │
│  │   (7B)       │  │                 │  │  (Function   │  │
│  └──────┬───────┘  │                 │  │   Calling)   │  │
│         │          │                 │  └──────┬───────┘  │
│         ▼          │                 │         │          │
│  Action Vectors    │                 │  Motor Primitives  │
│  (50Hz)            │                 │  + Speech          │
└────────┬───────────┘                 └────────┬───────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  ROBOT ACTIONS  │
              │ (Arm + Base +   │
              │   Gripper)      │
              └─────────────────┘
```

### Network Requirements
- **Bandwidth**:
  - π0 Mode: ~2 Mbps (30fps @ 224x224 RGB + action downlink)
  - Gemini Mode: ~0.5 Mbps (5fps + audio stream)
- **Latency**:
  - Target: <200ms RTT to Akamai server
  - Acceptable: <500ms (higher latency → jerky motions)
- **Reliability**: WiFi 5 (802.11ac) or Ethernet recommended

### Error Handling
- **Network Dropout**:
  - Buffer last 5 actions from π0, continue execution
  - If >2s disconnect → emergency stop
  - Cache Gemini responses for offline playback (stretch goal)
- **Model Failures**:
  - π0 returns invalid action → clamp to safe joint limits
  - Gemini timeout → repeat last query or return to idle
- **Hardware Failures**:
  - Camera disconnect → halt VLA mode, switch to safe teleoperation
  - Motor error → send alert, disable autonomous mode

## 7. Deployment & Setup

### Initial Setup Steps
1. **Flash Jetson Orin Nano**: JetPack 6.0 (Ubuntu 22.04)
2. **Install ROS2 Humble**: `apt install ros-humble-desktop`
3. **Install LeRobot SDK**: `pip install lerobot[lekiwi]`
4. **Configure Cameras**: RealSense SDK + camera calibration
5. **Deploy Akamai Server**: Docker image with π0 model
6. **API Keys**: Set `GEMINI_API_KEY` and `AKAMAI_VLA_TOKEN`
7. **Test Motors**: Run LeKiwi self-test and calibration
8. **Run Orchestration**: `python orchestrate.py --mode hybrid`

### Configuration Files
- `config/robot_config.yaml`: Hardware parameters, camera IDs, joint limits
- `config/vla_server.yaml`: Akamai endpoint, auth token, timeout settings
- `config/gemini_config.yaml`: API key, system prompt, function definitions
- `config/safety_limits.yaml`: Speed caps, workspace boundaries, collision thresholds
