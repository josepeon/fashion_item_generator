# Fashion Diffusion - SD LoRA Fine-tuning Roadmap

A production-grade text-to-fashion image generator using Stable Diffusion with LoRA fine-tuning.

## Project Overview

### What You'll Build
A model that actually understands prompts like:
- "bulky sneaker with chunky sole and mesh upper"
- "elegant black evening dress with lace details"
- "vintage denim jacket with distressed patches"

### How It Works
```
Text Prompt → CLIP Text Encoder → Cross-Attention → U-Net (LoRA) → Decoded Image
                                        ↑
                              Fashion-specific weights
```

### Why LoRA?
- Full SD fine-tuning: 4GB+ VRAM, 40GB+ disk, days of training
- LoRA fine-tuning: 8GB VRAM, ~100MB weights, hours of training
- Same quality, 100x more efficient

---

## Prerequisites

### Hardware Options

| Option | VRAM | Training Time | Cost |
|--------|------|---------------|------|
| **Local Mac M1/M2/M3** | 16-32GB unified | 24-48 hours | Free |
| **Local NVIDIA GPU** | 12GB+ (3060+) | 8-16 hours | Free |
| **Google Colab Pro** | 16GB (A100) | 2-4 hours | $10/mo |
| **RunPod/Vast.ai** | 24GB (A10/4090) | 1-2 hours | $0.50/hr |
| **Lambda Labs** | 24GB (A10) | 1-2 hours | $0.75/hr |

**Recommendation**: Start with Colab Pro to learn, then move to RunPod for serious training.

### Software Requirements
```bash
# Core
python>=3.10
torch>=2.0
diffusers>=0.25.0
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0  # LoRA implementation

# Training
datasets
wandb  # Experiment tracking
safetensors

# Data prep
pillow
albumentations  # Augmentation
```

---

## Phase 1: Dataset Preparation (Week 1)

### Dataset Options

| Dataset | Images | Resolution | Labels | Size |
|---------|--------|------------|--------|------|
| **DeepFashion** | 800K | 256×256 | Rich attributes | ~50GB |
| **Fashion Product Images** | 44K | 400×400 | Category + color | ~15GB |
| **iMaterialist Fashion** | 1M | Varies | Fine-grained | ~100GB |
| **Polyvore Outfits** | 365K | Varies | Outfit combos | ~30GB |
| **Your own scrape** | Custom | Custom | Custom | Varies |

### Recommended: DeepFashion-Inshop
- 52K images, well-curated
- Good attribute annotations
- Manageable size for learning

### Step 1.1: Download & Organize

```python
# scripts/download_dataset.py

import os
from pathlib import Path
from datasets import load_dataset

def download_fashion_dataset(output_dir: str = "data/fashion"):
    """Download and prepare fashion dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Option 1: HuggingFace dataset
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    
    # Save images with captions
    for i, item in enumerate(ds):
        img = item["image"]
        caption = f"{item['productDisplayName']}, {item['articleType']}, {item['baseColour']}"
        
        img_path = output_path / f"{i:06d}.png"
        caption_path = output_path / f"{i:06d}.txt"
        
        img.save(img_path)
        caption_path.write_text(caption)
        
        if i % 1000 == 0:
            print(f"Processed {i} images...")
    
    print(f"Done! {i+1} images saved to {output_path}")


if __name__ == "__main__":
    download_fashion_dataset()
```

### Step 1.2: Generate Captions (if needed)

```python
# scripts/generate_captions.py

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def generate_captions(image_dir: str, output_dir: str = None):
    """Generate captions for images using BLIP."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    
    image_path = Path(image_dir)
    output_path = Path(output_dir or image_dir)
    
    for img_file in tqdm(list(image_path.glob("*.png")) + list(image_path.glob("*.jpg"))):
        caption_file = output_path / f"{img_file.stem}.txt"
        
        if caption_file.exists():
            continue
        
        image = Image.open(img_file).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate with fashion-specific prompt
        out = model.generate(
            **inputs,
            max_length=50,
            num_beams=4,
        )
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Enhance caption with fashion prefix
        caption = f"a photo of {caption}, fashion product, studio lighting"
        caption_file.write_text(caption)


if __name__ == "__main__":
    generate_captions("data/fashion")
```

### Step 1.3: Create Training Dataset

```python
# scripts/prepare_dataset.py

from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

def prepare_training_data(
    image_dir: str,
    output_file: str = "data/train_data.jsonl",
    image_size: int = 512
):
    """Prepare dataset in format expected by diffusers."""
    image_path = Path(image_dir)
    output_path = Path(output_file)
    
    records = []
    
    for img_file in tqdm(list(image_path.glob("*.png")) + list(image_path.glob("*.jpg"))):
        caption_file = image_path / f"{img_file.stem}.txt"
        
        if not caption_file.exists():
            continue
        
        # Validate image
        try:
            img = Image.open(img_file)
            if img.size[0] < 256 or img.size[1] < 256:
                continue
        except:
            continue
        
        caption = caption_file.read_text().strip()
        
        records.append({
            "image": str(img_file.absolute()),
            "text": caption,
        })
    
    # Write JSONL
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"Created {len(records)} training examples")
    return records


if __name__ == "__main__":
    prepare_training_data("data/fashion")
```

---

## Phase 2: LoRA Training (Week 2)

### Step 2.1: Training Script

```python
# train_lora.py

import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from pathlib import Path
import wandb

# ============ Configuration ============

MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # or "stabilityai/stable-diffusion-2-1"
OUTPUT_DIR = "output/fashion-lora"
TRAIN_DATA = "data/train_data.jsonl"

# LoRA config
LORA_RANK = 32  # Higher = more capacity, more VRAM
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training config
BATCH_SIZE = 1  # Increase if you have VRAM
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
SAVE_STEPS = 500

# ============ Setup ============

def setup_lora_unet(unet):
    """Add LoRA layers to U-Net."""
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "to_q", "to_v", "to_k", "to_out.0",  # Attention
            "proj_in", "proj_out",  # Projections
            "ff.net.0.proj", "ff.net.2",  # FFN
        ],
    )
    return get_peft_model(unet, lora_config)


def train():
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        mixed_precision="fp16",
        log_with="wandb",
    )
    
    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(project="fashion-lora", name="sd-fashion-v1")
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )
    unet = pipe.unet
    vae = pipe.vae
    
    # Freeze everything except LoRA
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Add LoRA
    unet = setup_lora_unet(unet)
    unet.print_trainable_parameters()  # Should show ~0.1% trainable
    
    # Load dataset
    dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    
    # Preprocessing
    def preprocess(examples):
        images = [Image.open(p).convert("RGB").resize((512, 512)) for p in examples["image"]]
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        images = images * 2 - 1  # Normalize to [-1, 1]
        
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        
        return {"pixel_values": images, "input_ids": tokens.input_ids}
    
    dataset = dataset.map(preprocess, batched=True, batch_size=BATCH_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    
    # Prepare for distributed
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    # Training loop
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        unet.train()
        
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Encode images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
                
                # Add noise to latents
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % 100 == 0:
                accelerator.log({"loss": loss.item()}, step=global_step)
                print(f"Step {global_step}, Loss: {loss.item():.4f}")
            
            if global_step % SAVE_STEPS == 0:
                save_lora(unet, f"{OUTPUT_DIR}/checkpoint-{global_step}")
    
    # Save final
    save_lora(unet, f"{OUTPUT_DIR}/final")


def save_lora(model, path):
    """Save only LoRA weights."""
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    print(f"Saved LoRA weights to {path}")


if __name__ == "__main__":
    train()
```

### Step 2.2: Simplified Training with diffusers CLI

For easier start, use the built-in trainer:

```bash
# train_lora.sh

accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="data/fashion" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --output_dir="output/fashion-lora" \
  --validation_prompt="a stylish black leather jacket with silver zippers" \
  --validation_epochs=10 \
  --seed=42 \
  --mixed_precision="fp16" \
  --checkpointing_steps=500 \
  --report_to="wandb"
```

---

## Phase 3: Inference & API (Week 3)

### Step 3.1: Load and Use LoRA

```python
# src/inference.py

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel

class FashionDiffusion:
    """Fashion image generation with LoRA fine-tuned Stable Diffusion."""
    
    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        lora_path: str = "output/fashion-lora/final",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load base model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            safety_checker=None,
        )
        
        # Use faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Load LoRA weights
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            lora_path,
        )
        
        self.pipe.to(self.device)
        
        # Optimizations
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing()
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        num_images: int = 1,
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,
    ):
        """Generate fashion images from prompt."""
        
        # Add fashion-specific prompt engineering
        enhanced_prompt = f"{prompt}, fashion product photography, studio lighting, high quality, detailed"
        
        generator = torch.Generator(self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        images = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images
        
        return images
    
    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        **kwargs
    ):
        """Generate multiple variations with different seeds."""
        import random
        
        images = []
        for _ in range(num_variations):
            seed = random.randint(0, 2**32 - 1)
            img = self.generate(prompt, num_images=1, seed=seed, **kwargs)[0]
            images.append(img)
        
        return images


# Usage
if __name__ == "__main__":
    model = FashionDiffusion()
    
    images = model.generate(
        prompt="bulky sneaker with chunky sole and mesh upper, white and grey",
        num_images=4,
    )
    
    for i, img in enumerate(images):
        img.save(f"output/sneaker_{i}.png")
```

### Step 3.2: FastAPI Endpoints

```python
# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import base64
import io

from inference import FashionDiffusion

app = FastAPI(title="Fashion Diffusion API", version="1.0.0")

# Load model on startup
model: FashionDiffusion = None

@app.on_event("startup")
async def load_model():
    global model
    model = FashionDiffusion()
    print("✓ Fashion Diffusion model loaded")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the fashion item")
    negative_prompt: Optional[str] = "blurry, low quality, distorted"
    num_images: int = Field(default=1, ge=1, le=4)
    num_steps: int = Field(default=25, ge=10, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    prompt: str
    images: list[str]  # Base64 encoded PNGs


def pil_to_base64(img) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate fashion images from text prompt."""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    try:
        images = model.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_images=request.num_images,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        
        return GenerateResponse(
            prompt=request.prompt,
            images=[pil_to_base64(img) for img in images],
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## Phase 4: Web Interface (Week 4)

### Step 4.1: Streamlit App

```python
# app.py

import streamlit as st
import requests
from PIL import Image
import io
import base64

API_URL = "http://localhost:8080"

st.set_page_config(
    page_title="Fashion Diffusion",
    page_icon="👗",
    layout="wide",
)

st.title("👗 Fashion Diffusion")
st.markdown("Generate fashion items from text descriptions")

# Prompt input
col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area(
        "Describe the fashion item you want to create:",
        placeholder="e.g., 'bulky sneaker with chunky sole and mesh upper, white and grey colorway'",
        height=100,
    )

with col2:
    num_images = st.slider("Number of images", 1, 4, 2)
    guidance = st.slider("Creativity", 1.0, 15.0, 7.5)
    steps = st.slider("Quality (steps)", 10, 50, 25)

# Advanced options
with st.expander("Advanced Options"):
    negative_prompt = st.text_input(
        "Negative prompt (what to avoid):",
        value="blurry, low quality, distorted, deformed, ugly",
    )
    seed = st.number_input("Seed (0 for random)", min_value=0, value=0)

# Generate button
if st.button("✨ Generate", type="primary", disabled=not prompt):
    with st.spinner("Creating your fashion item..."):
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_images": num_images,
                "num_steps": steps,
                "guidance_scale": guidance,
                "seed": seed if seed > 0 else None,
            },
        )
        
        if response.status_code == 200:
            data = response.json()
            
            cols = st.columns(num_images)
            for i, img_b64 in enumerate(data["images"]):
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes))
                
                with cols[i]:
                    st.image(img, caption=f"Variation {i+1}", use_column_width=True)
                    
                    # Download button
                    st.download_button(
                        f"💾 Download",
                        img_bytes,
                        f"fashion_{i+1}.png",
                        "image/png",
                    )
        else:
            st.error(f"Error: {response.text}")

# Example prompts
st.markdown("---")
st.markdown("### Example Prompts")

examples = [
    "sleek minimalist white sneaker with clean lines",
    "vintage denim jacket with distressed patches and brass buttons",
    "elegant black evening dress with lace details and flowing silhouette",
    "oversized hoodie in earth tones with embroidered logo",
    "structured leather handbag with gold hardware and quilted pattern",
]

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    with cols[i]:
        if st.button(f"Try", key=f"example_{i}"):
            st.session_state.prompt = example
            st.rerun()
```

---

## Phase 5: Deployment (Week 5)

### Cloud GPU Options for Inference

| Service | GPU | Cost | Setup |
|---------|-----|------|-------|
| **RunPod Serverless** | A10/A100 | $0.00025/sec | Easy |
| **Replicate** | A40 | $0.00055/sec | Easiest |
| **Modal** | A10G | $0.000306/sec | Easy |
| **AWS Lambda + EFS** | Custom | Variable | Complex |
| **Hugging Face Inference** | T4/A10G | $0.06/hr | Easy |

### Recommended: Modal or Replicate

#### Modal Deployment

```python
# modal_app.py

import modal

app = modal.App("fashion-diffusion")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
)

# Create a volume for model weights
volume = modal.Volume.from_name("fashion-lora-weights", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/weights": volume},
    container_idle_timeout=300,
)
class FashionDiffusion:
    @modal.enter()
    def load_model(self):
        from inference import FashionDiffusion
        self.model = FashionDiffusion(lora_path="/weights/fashion-lora")
    
    @modal.method()
    def generate(self, prompt: str, num_images: int = 1):
        images = self.model.generate(prompt, num_images=num_images)
        return [img_to_base64(img) for img in images]


@app.function(image=image)
@modal.web_endpoint(method="POST")
def generate_endpoint(prompt: str, num_images: int = 1):
    model = FashionDiffusion()
    return {"images": model.generate.remote(prompt, num_images)}
```

---

## Project Structure

```
fashion_diffusion/
├── data/
│   ├── fashion/              # Training images + captions
│   └── train_data.jsonl      # Prepared dataset
├── scripts/
│   ├── download_dataset.py
│   ├── generate_captions.py
│   └── prepare_dataset.py
├── src/
│   ├── inference.py          # FashionDiffusion class
│   ├── api.py                # FastAPI endpoints
│   └── app.py                # Streamlit interface
├── output/
│   └── fashion-lora/         # Trained LoRA weights
├── train_lora.py             # Training script
├── train_lora.sh             # Training with diffusers CLI
├── modal_app.py              # Modal deployment
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Dataset | 10K+ captioned fashion images ready for training |
| 2 | Training | LoRA weights that generate fashion items |
| 3 | Inference | API that takes prompts, returns images |
| 4 | Frontend | Streamlit app for interactive generation |
| 5 | Deploy | Live API on Modal/Replicate |

---

## Builds On Fashion-MNIST Project

| Skill | Fashion-MNIST | Fashion Diffusion |
|-------|---------------|-------------------|
| CNN architecture | ✅ Built | → Understanding helps debug U-Net |
| VAE latent space | ✅ Built | → Diffusion uses similar latent concepts |
| Training loops | ✅ Built | → Same patterns, larger scale |
| FastAPI | ✅ Built | → Same patterns, GPU inference |
| Streamlit | ✅ Built | → Same patterns, image handling |
| Deployment | ✅ Streamlit Cloud | → GPU cloud (Modal, Replicate) |

---

## Cost Estimate

### Training (one-time)
- Colab Pro: $10/mo (plenty for learning)
- RunPod: ~$5-10 for full training run

### Inference (ongoing)
- Modal: ~$50/mo for moderate usage (1000 images/day)
- Replicate: ~$30/mo for moderate usage
- Self-hosted (if you have GPU): Free

---

## Next Steps

1. **Create new repo**: `fashion_diffusion`
2. **Start with dataset**: Download DeepFashion or Fashion Product Images
3. **Train on Colab**: Use free GPU to learn the process
4. **Iterate**: Improve prompts, retrain, test
5. **Deploy**: Modal or Replicate for production

---

## Resources

### Tutorials
- [HuggingFace LoRA Training Guide](https://huggingface.co/docs/diffusers/training/lora)
- [Kohya LoRA Trainer](https://github.com/kohya-ss/sd-scripts) (popular alternative)
- [Modal Stable Diffusion Example](https://modal.com/docs/examples/stable_diffusion)

### Datasets
- [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- [Fashion Product Images (Kaggle)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- [iMaterialist Fashion](https://github.com/visipedia/imat_fashion_comp)

### Papers
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752)
- [DreamBooth](https://arxiv.org/abs/2208.12242)
