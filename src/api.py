#!/usr/bin/env python3
"""FastAPI endpoints for Fashion-MNIST inference."""

import io
import base64
from typing import Optional

import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models import FashionCNN, FashionVAE

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Class names
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Initialize app
app = FastAPI(
    title="Fashion-MNIST API",
    description="Classification and generation for fashion items",
    version="1.0.0",
)

# Global models (loaded once)
cnn: Optional[FashionCNN] = None
vae: Optional[FashionVAE] = None
device: torch.device = None


def get_device() -> torch.device:
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global cnn, vae, device
    device = get_device()
    
    # Load CNN
    cnn = FashionCNN()
    cnn_path = WEIGHTS_DIR / "cnn.pth"
    try:
        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        cnn.to(device).eval()
        print(f"✓ CNN loaded on {device}")
    except FileNotFoundError:
        print(f"⚠ CNN weights not found at {cnn_path}")
        cnn = None
    
    # Load VAE
    vae = FashionVAE(latent_dim=32)
    vae_path = WEIGHTS_DIR / "vae.pth"
    try:
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        vae.to(device).eval()
        print(f"✓ VAE loaded on {device}")
    except FileNotFoundError:
        print(f"⚠ VAE weights not found at {vae_path}")
        vae = None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for CNN inference."""
    # Convert to grayscale and resize
    image = image.convert("L").resize((28, 28))
    # Normalize to [0, 1] and add batch/channel dims
    tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor to base64 PNG string."""
    # Denormalize and convert to uint8
    img_array = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="L")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# Request/Response models
class ClassifyResponse(BaseModel):
    class_id: int = Field(..., description="Predicted class index (0-9)")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: dict = Field(..., description="All class probabilities")


class GenerateRequest(BaseModel):
    class_id: int = Field(..., ge=0, le=9, description="Class to generate (0-9)")
    num_samples: int = Field(default=4, ge=1, le=16, description="Number of samples")


class GenerateResponse(BaseModel):
    class_name: str
    images: list[str] = Field(..., description="Base64-encoded PNG images")


class HealthResponse(BaseModel):
    status: str
    cnn_loaded: bool
    vae_loaded: bool
    device: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health and model status."""
    return HealthResponse(
        status="ok",
        cnn_loaded=cnn is not None,
        vae_loaded=vae is not None,
        device=str(device),
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(..., description="28x28 grayscale image")):
    """Classify a fashion item image."""
    if cnn is None:
        raise HTTPException(503, "CNN model not loaded")
    
    # Read and preprocess image
    try:
        image = Image.open(io.BytesIO(await file.read()))
        tensor = preprocess_image(image).to(device)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    
    # Inference
    with torch.no_grad():
        logits = cnn(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        class_id = probs.argmax().item()
        confidence = probs[class_id].item()
    
    return ClassifyResponse(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        confidence=round(confidence, 4),
        probabilities={CLASS_NAMES[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate fashion item images for a given class."""
    if vae is None:
        raise HTTPException(503, "VAE model not loaded")
    
    # Generate samples
    with torch.no_grad():
        samples = vae.generate(request.num_samples, request.class_id, device)
    
    # Convert to base64
    images = [tensor_to_base64(samples[i]) for i in range(request.num_samples)]
    
    return GenerateResponse(
        class_name=CLASS_NAMES[request.class_id],
        images=images,
    )


@app.get("/classes")
async def list_classes():
    """List all available fashion classes."""
    return {i: name for i, name in enumerate(CLASS_NAMES)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
