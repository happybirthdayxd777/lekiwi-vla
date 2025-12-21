# Lekiwi Hybrid VLA Robot

A dual-mode intelligent robotic system combining Vision-Language-Action (VLA) models with Large Language Model (LLM) agents for precise manipulation and autonomous exploration.

## Overview

Lekiwi is built on the LeKiwi (SO-101) platform and features a hybrid AI architecture:

- **Worker Mode (OpenPI/π0)**: Precision manipulation using a 7B Vision-Language-Action model
- **Explorer Mode (Gemini)**: Autonomous exploration and natural interaction using LLM-based planning

### Core Philosophy
**"Local Hearing, Cloud Thinking, Hybrid Action"**

## Hardware

- **Robot**: LeKiwi (SO-101) - 6-DOF robotic arm with mobile base
- **Edge Compute**: NVIDIA Jetson Orin Nano Super (8GB)
- **Cloud GPU**: Akamai Cloud Instance with RTX 4000 Ada (20GB VRAM)
- **Sensors**: RealSense depth camera, dual RGB cameras, IMU
- **Audio**: USB microphone and speaker for voice interaction

## Key Features

- **Dual-Mode Operation**: Seamlessly switch between precision VLA and exploratory LLM modes
- **Automatic Fallback**: Intelligent mode switching based on task requirements
- **Voice Control**: Local speech recognition with Whisper
- **Visual SLAM**: Real-time mapping and localization
- **Function Calling**: xLeRobot-style LLM agent control with motor primitives
- **Cloud-Edge Hybrid**: Distributed processing for optimal performance

## Architecture

```
Voice Input → Jetson (Whisper STT) → Command
                    ↓
Camera Stream → Akamai GPU (π0 VLA) → Action Chunks → Jetson → Motors
                    OR
Audio/Video → Gemini API → Function Calls → Jetson → Motors
```

## Documentation

- **[requirements.md](requirements.md)**: Complete technical requirements and specifications
- **[implementation_plan.md](implementation_plan.md)**: Implementation architecture and proposed changes
- **[operations_guide.md](operations_guide.md)**: Setup and operational procedures
- **[task.md](task.md)**: Project task tracking

## Quick Start

### Prerequisites
- Jetson Orin Nano with JetPack 6.0 (Ubuntu 22.04)
- ROS2 Humble
- Akamai Cloud GPU instance
- Google Gemini API key

### Installation

See [operations_guide.md](operations_guide.md) for detailed setup instructions.

## References

- [LeKiwi Official Documentation](https://huggingface.co/docs/lerobot/en/lekiwi)
- [xLeRobot LLM Agent Control](https://xlerobot.readthedocs.io/en/latest/software/getting_started/LLM_agent.html)
- [OpenPI VLA Model](https://github.com/physical-intelligence/pi0)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)

## License

TBD

## Contributing

TBD
