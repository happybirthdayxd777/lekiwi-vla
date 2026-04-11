#!/usr/bin/env python3
"""
Pi0 VLA Server for LeKiwi Robot
Cloud-side inference endpoint using FastAPI + LeRobot

Handles /act endpoint that receives (image, text) and returns action tensor.
"""

import io
import base64
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image

# LeRobot imports - these would be installed from the lekiwi-vla repo
import sys
sys.path.insert(0, "/Users/i_am_ai/lerobot/src")

from lerobot.policies.pi0.agent import Pi0Agent
from lerobot.policies.pi0.modeling_pi0 import Pi0Config


class ActRequest(BaseModel):
    image_base64: str
    text: str
    robot_state: Optional[dict] = None


class ActResponse(BaseModel):
    action: list[float]
    policy_time_ms: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Pi0 model at startup."""
    print("Loading Pi0 model...")
    try:
        # Config for Pi0-7B
        config = Pi0Config(
            vision_encoder="siglip",
            vision_encoder_pretrained="google/siglip-base-patch16-224",
            language_model="meta-llama/Llama-3.2-3B-Instruct",
            action_dim=9,  # 6 arm + 3 base joints
        )
        
        # In production, load from HuggingFace Hub:
        # agent = Pi0Agent.from_pretrained("PhysicalAI/pi0-7b")
        # For now, just print config
        print(f"Pi0 config created: action_dim={config.action_dim}")
        
        app.state.agent = None  # Placeholder
        app.state.model_loaded = False
        
    except Exception as e:
        print(f"Failed to load Pi0: {e}")
        app.state.model_loaded = False
    
    yield
    
    print("Shutting down Pi0 server...")


app = FastAPI(title="Pi0 VLA Server", lifespan=lifespan)


@app.post("/act", response_model=ActResponse)
async def act(request: ActRequest):
    """Main VLA inference endpoint."""
    import time
    start = time.time()
    
    try:
        # Decode image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # In production:
        # action = await app.state.agent.predict(image, request.text)
        
        # Placeholder: return zero action for testing
        action = [0.0] * 9  # 6 arm + 3 base
        
        elapsed_ms = (time.time() - start) * 1000
        
        return ActResponse(
            action=action,
            policy_time_ms=round(elapsed_ms, 2),
            model_version="pi0-7b-placeholder"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model_loaded", False),
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Pi0 VLA Server",
        "version": "0.1.0",
        "endpoints": ["/act", "/health"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)