# Implementation Plan - LeKiwi Hybrid System

## Goal Description
Deploy the **Dual-Mode Architecture** on LeKiwi hardware.
1.  **Server**: Pi0 inference deployed on **Akamai Cloud (RTX 4000 Ada)**.
2.  **Client**: Jetson orchestrator managing Whisper, VSLAM, and Mode Switching.

## User Review Required
> [!IMPORTANT]
> **Cloud Deployment**: The `vla_server.py` MUST be deployed to the Akamai instance (not run locally) to support the 7B model.
> **Jetson Optimization**: Running Whisper, VSLAM, and Camera Streams on 8GB shared memory is the bottleneck. use `zram`.

## Proposed Changes

### 1. Cloud Server (Akamai Deployment)
*File: `server/`*

#### [NEW] [vla_server.py](file:///Users/chenshi/.gemini/antigravity/brain/ed960d05-c966-4ef9-acf5-e4c680a9bd88/server/vla_server.py)
- **Dependencies**: `lerobot` (for Pi0), `transformers`, `fastapi`.
- **Endpoints**:
  - `POST /act`: Input `(image, text)` -> Output `action_tensor` (Pi0).

### 2. Edge Client (Jetson Orin)
*File: `client/`*

#### [NEW] [main_client.py](file:///Users/chenshi/.gemini/antigravity/brain/ed960d05-c966-4ef9-acf5-e4c680a9bd88/client/main_client.py)
- **The Orchestrator**: Init drivers, start threads.
- **Keyboard Listener**: Toggles `self.mode = 'WORKER' | 'EXPLORER'`.
- **Loop**:
  - Reads Sensors (LeKiwi Arm, Cameras).
  - Routes data to `WorkerAgent` or `ExplorerAgent`.

#### [NEW] [worker_agent.py](file:///Users/chenshi/.gemini/antigravity/brain/ed960d05-c966-4ef9-acf5-e4c680a9bd88/client/worker_agent.py)
- **STT**: Calls `audio_service.transcribe()`.
- **Logic**:
  1. Detect Command ("Grab").
  2. Entering `Pi0_Loop`: 
     - Stream Head/Wrist frames + Text to `/act` (on Akamai).
     - Receive `action_delta`.
     - Execute via `hardware_interface`.

#### [NEW] [explorer_agent.py](file:///Users/chenshi/.gemini/antigravity/brain/ed960d05-c966-4ef9-acf5-e4c680a9bd88/client/explorer_agent.py)
- **Gemini Config**:
  - Define `tools = [move_forward, turn_left, turn_right, stop]`.
- **Loop**:
  - Sends buffered audio/video.
  - Receives `part` with `function_call`.
  - Executes function `move_forward()` on `hardware_interface`.
  - Sends back `function_response`.

#### [NEW] [hardware_interface.py](file:///Users/chenshi/.gemini/antigravity/brain/ed960d05-c966-4ef9-acf5-e4c680a9bd88/client/hardware_interface.py)
- **Dependencies**: `lerobot.common.robot_devices.robots.factory`.
- **Logic**:
  - `self.robot = make_robot("lekiwi")` (Automatic Feetech SDK init).
  - `connect()`: Connects to Motor Bus.
  - `move_base(x, y)`: Maps unified actions to Wheel IDs (7,8,9).
  - `VSLAM Integration`: Fuses `lerobot` motor odometry with Isaac ROS visual odometry.

## Verification Plan
1.  **Cloud Check**: Verify Akamai IP is reachable and `vla_server` is responding.
2.  **Hardware Test**: Verify LeKiwi base moves 1 meter when command `move(1.0)` is sent.
3.  **Integration**: "Switch Mode" toggle works instantly without crashing threads.
