#!/usr/bin/env python3
"""
VLA Inference Demo
=================
Demonstrates VLA (Vision-Language-Action) inference pipeline
using dummy data. Shows the complete flow from text command
to robot action.

Can run on MacBook without real robot.

Usage:
    python3 vla_demo.py --mode server     # Run as API server
    python3 vla_demo.py --mode client      # Run as robot client
    python3 vla_demo.py --mode sim         # Run simulation
"""

import argparse
import json
import time
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Simulated VLA model (placeholder)
MOCK_MODEL_RESPONSE = {
    "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],  # 9-DOF
    "reasoning": "The robot should move forward slightly to approach the object",
    "confidence": 0.85,
}


def generate_dummy_image(text: str = "robot view") -> Image.Image:
    """Generate a dummy camera image."""
    img = Image.new('RGB', (640, 480), color=(50, 50, 80))
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes to simulate robot view
    draw.rectangle([100, 100, 540, 380], outline='white', width=2)
    draw.text((200, 200), f"Robot View", fill='white')
    draw.text((200, 250), f"Text: {text[:30]}...", fill='lightgray')
    draw.text((200, 300), "Objects: [red box, blue cylinder]", fill='gray')
    
    # Draw some "detected objects"
    draw.rectangle([150, 150, 200, 200], outline='red', width=3)
    draw.ellipse([350, 150, 420, 220], outline='blue', width=3)
    
    return img


def simulate_vla_inference(image: Image.Image, text: str) -> dict:
    """
    Simulates VLA model inference.
    In production, this would call Pi0/Pi0-ext/OpenVLA.
    """
    # Simulate processing time
    time.sleep(0.1)
    
    # Parse the command
    text_lower = text.lower()
    
    if "forward" in text_lower:
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]
        reasoning = "Moving robot forward"
    elif "backward" in text_lower:
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0]
        reasoning = "Moving robot backward"
    elif "left" in text_lower:
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        reasoning = "Turning robot left"
    elif "right" in text_lower:
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5]
        reasoning = "Turning robot right"
    elif "grab" in text_lower or "pick" in text_lower:
        action = [0.0, 0.5, -0.3, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
        reasoning = "Moving arm to grab object"
    elif "stop" in text_lower:
        action = [0.0] * 9
        reasoning = "Stopping all motion"
    else:
        action = [0.0] * 9
        reasoning = f"Unknown command '{text}', stopping for safety"
    
    return {
        "action": action,
        "reasoning": reasoning,
        "confidence": 0.9,
    }


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_image(b64: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data))


class RobotSimulator:
    """Simple robot state simulator for demo."""
    
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.arm = [0.0] * 6
        
    def apply_action(self, action):
        """Apply 9-DOF action to robot state."""
        # action[0:6] = arm joint positions
        # action[6:9] = base velocity (vx, vy, wz)
        self.arm = list(action[:6])
        vx, vy, wz = action[6], action[7], action[8]
        
        # Simple kinematic update
        self.x += vx * 0.1
        self.y += vy * 0.1
        self.theta += wz * 0.1
        
    def render(self, text: str) -> Image.Image:
        """Render current robot state as image."""
        img = generate_dummy_image(text)
        draw = ImageDraw.Draw(img)
        
        # Draw robot state overlay
        state_text = f"Position: x={self.x:.2f} y={self.y:.2f} θ={self.theta:.2f}"
        draw.text((20, 20), state_text, fill='lime')
        
        arm_str = f"Arm: {', '.join([f'{a:.1f}' for a in self.arm[:3]])}"
        draw.text((20, 40), arm_str, fill='lime')
        
        return img


def run_simulation():
    """Run VLA demo in simulation mode (no real hardware)."""
    print("=" * 60)
    print("  VLA Demo - Simulation Mode")
    print("=" * 60)
    print()
    
    robot = RobotSimulator()
    
    # Test commands
    test_commands = [
        "move forward",
        "turn left",
        "grab the red box",
        "stop",
        "move backward",
        "turn right",
    ]
    
    for cmd in test_commands:
        print(f"\n[Command]: {cmd}")
        
        # Generate image (simulating robot camera)
        img = robot.render(cmd)
        
        # Run VLA inference
        result = simulate_vla_inference(img, cmd)
        
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Action: {[f'{a:.2f}' for a in result['action']]}")
        
        # Apply action to robot
        robot.apply_action(result['action'])
        print(f"  New position: x={robot.x:.2f} y={robot.y:.2f} θ={robot.theta:.2f}")
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("Simulation complete!")


def run_server_demo(port=8000):
    """Run VLA API server demo."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/act', methods=['POST'])
    def act():
        data = request.json
        image_b64 = data.get('image_base64', '')
        text = data.get('text', '')
        
        # Decode image if provided
        if image_b64:
            img = base64_to_image(image_b64)
        else:
            img = generate_dummy_image(text)
        
        # Run VLA inference
        result = simulate_vla_inference(img, text)
        
        return jsonify({
            'action': result['action'],
            'reasoning': result['reasoning'],
            'confidence': result['confidence'],
            'processing_time_ms': 100,
        })
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'model': 'pi0-placeholder'})
    
    print(f"Starting VLA server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)


def run_client_demo():
    """Run VLA client demo - shows how to call the API."""
    print("VLA Client Demo")
    print("-" * 40)
    
    # Generate dummy image
    img = generate_dummy_image("pick up the red object")
    
    # Simulate API call (in production, use requests.post)
    print("Sending request to /act endpoint...")
    
    result = simulate_vla_inference(img, "pick up the red object")
    
    print(f"\nResponse:")
    print(f"  Action: {result['action']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    
    # Show action interpretation
    arm_action = result['action'][:6]
    base_action = result['action'][6:]
    print(f"\nInterpretation:")
    print(f"  Arm targets: {[f'{a:.2f}' for a in arm_action]}")
    print(f"  Base velocity: {base_action}")


def main():
    parser = argparse.ArgumentParser(description="VLA Inference Demo")
    parser.add_argument(
        '--mode',
        choices=['sim', 'server', 'client'],
        default='sim',
        help='Run mode: sim (default), server, or client'
    )
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    args = parser.parse_args()
    
    if args.mode == 'sim':
        run_simulation()
    elif args.mode == 'server':
        run_server_demo(args.port)
    elif args.mode == 'client':
        run_client_demo()


if __name__ == '__main__':
    main()