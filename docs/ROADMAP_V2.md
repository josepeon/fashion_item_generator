# Fashion Item Generator V2 - Interactive Web Platform

## Vision

A web platform where users can:
1. Generate fashion items from text prompts
2. Use existing features (classify, generate, interpolate, style transfer)
3. Every generated image gets added to a growing "chain" of interpolations
4. The chain forms a live, ever-growing background mosaic

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────────┐   │
│  │ Prompt  │ │Generate │ │Interpolate│ │ Live Background    │   │
│  │ Input   │ │ Button  │ │ Controls │ │ (WebSocket updates)│   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │ /generate   │ │ /chain      │ │ Background Worker       │   │
│  │ from prompt │ │ endpoints   │ │ (interpolation queue)   │   │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ PostgreSQL│   │ Redis    │   │ S3/Cloud │
        │ (metadata)│   │ (queue)  │   │ (images) │
        └──────────┘   └──────────┘   └──────────┘
```

---

## Phase 1: Text-to-Fashion Generation

### The Challenge

Current VAE generates from class IDs (0-9). We need to map text prompts to generations.

### Approach Options

| Approach | Complexity | Quality | Training Required |
|----------|------------|---------|-------------------|
| **A. Keyword Mapping** | Low | Medium | No |
| **B. CLIP + VAE** | Medium | High | Fine-tuning |
| **C. Stable Diffusion LoRA** | High | Best | Yes |

### Recommended: Start with A, evolve to B

#### Step 1.1: Keyword-to-Class Mapping (Quick Win)

```python
# src/prompt_parser.py

import re
from typing import Tuple, Optional

CLASS_KEYWORDS = {
    0: ["t-shirt", "tshirt", "tee", "top", "tank"],
    1: ["trouser", "trousers", "pants", "jeans", "slacks"],
    2: ["pullover", "sweater", "sweatshirt", "hoodie", "jumper"],
    3: ["dress", "gown", "frock", "sundress"],
    4: ["coat", "jacket", "blazer", "overcoat", "parka"],
    5: ["sandal", "sandals", "flip-flop", "slides"],
    6: ["shirt", "blouse", "button-up", "oxford"],
    7: ["sneaker", "sneakers", "trainers", "running shoe", "athletic"],
    8: ["bag", "handbag", "purse", "backpack", "tote"],
    9: ["boot", "boots", "ankle boot", "bootie", "chelsea"],
}

STYLE_MODIFIERS = {
    "casual": {"temperature": 0.8},
    "formal": {"temperature": 0.5},
    "wild": {"temperature": 1.2},
    "minimal": {"temperature": 0.6},
    "experimental": {"temperature": 1.5},
}

def parse_prompt(prompt: str) -> Tuple[Optional[int], dict]:
    """
    Parse a text prompt into class ID and generation parameters.
    
    Examples:
        "a casual sneaker" → (7, {"temperature": 0.8})
        "formal dress" → (3, {"temperature": 0.5})
        "wild experimental bag" → (8, {"temperature": 1.5})
    
    Returns:
        (class_id, params) or (None, {}) if no match
    """
    prompt_lower = prompt.lower()
    
    # Find class
    detected_class = None
    for class_id, keywords in CLASS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                detected_class = class_id
                break
        if detected_class is not None:
            break
    
    # Find style modifiers
    params = {"temperature": 1.0}
    for style, style_params in STYLE_MODIFIERS.items():
        if style in prompt_lower:
            params.update(style_params)
            break
    
    return detected_class, params


def suggest_prompt(partial: str) -> list[str]:
    """Autocomplete suggestions for prompts."""
    suggestions = []
    partial_lower = partial.lower()
    
    for class_id, keywords in CLASS_KEYWORDS.items():
        for keyword in keywords:
            if keyword.startswith(partial_lower):
                suggestions.append(keyword)
    
    return suggestions[:5]
```

#### Step 1.2: Update API with Prompt Endpoint

```python
# Add to src/api.py

from prompt_parser import parse_prompt, suggest_prompt

class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Text description of fashion item")
    num_samples: int = Field(default=4, ge=1, le=8)

class PromptResponse(BaseModel):
    prompt: str
    detected_class: Optional[str]
    confidence: str  # "exact", "inferred", "random"
    images: list[str]  # base64


@app.post("/generate/prompt", response_model=PromptResponse)
async def generate_from_prompt(request: PromptRequest):
    """Generate fashion items from a text prompt."""
    if vae is None:
        raise HTTPException(503, "VAE model not loaded")
    
    class_id, params = parse_prompt(request.prompt)
    
    if class_id is None:
        # Random class if no match
        class_id = torch.randint(0, 10, (1,)).item()
        confidence = "random"
    else:
        confidence = "exact"
    
    with torch.no_grad():
        samples = vae.generate_class(
            class_id, 
            request.num_samples, 
            device,
            temperature=params.get("temperature", 1.0)
        )
    
    images = [tensor_to_base64(s) for s in samples]
    
    return PromptResponse(
        prompt=request.prompt,
        detected_class=CLASS_NAMES[class_id],
        confidence=confidence,
        images=images,
    )


@app.get("/suggest")
async def suggest(q: str = ""):
    """Get prompt autocomplete suggestions."""
    return {"suggestions": suggest_prompt(q)}
```

---

## Phase 2: Database & Image Storage

### Step 2.1: Database Schema

```sql
-- schema.sql

CREATE TABLE generated_images (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    prompt TEXT,
    class_id INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    temperature FLOAT DEFAULT 1.0,
    image_url TEXT NOT NULL,  -- S3/Cloudinary URL
    user_session VARCHAR(100),  -- Anonymous session ID
    is_in_chain BOOLEAN DEFAULT FALSE
);

CREATE TABLE chain_links (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    image_a_id INTEGER REFERENCES generated_images(id),
    image_b_id INTEGER REFERENCES generated_images(id),
    interpolation_url TEXT NOT NULL,  -- URL to interpolation GIF/frames
    position INTEGER NOT NULL  -- Order in the chain
);

CREATE INDEX idx_chain_position ON chain_links(position);
CREATE INDEX idx_images_created ON generated_images(created_at DESC);
```

### Step 2.2: Image Storage Service

```python
# src/storage.py

import boto3
import io
from PIL import Image
from datetime import datetime
from typing import Optional

class ImageStorage:
    """Store generated images in S3-compatible storage."""
    
    def __init__(self, bucket: str, endpoint_url: Optional[str] = None):
        self.bucket = bucket
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,  # For Cloudflare R2, MinIO, etc.
        )
    
    def save_image(self, tensor, prefix: str = "generated") -> str:
        """Save tensor as PNG, return URL."""
        # Convert tensor to PIL
        img_array = (tensor.squeeze().cpu().numpy() * 255).astype('uint8')
        img = Image.fromarray(img_array, mode='L')
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key = f"{prefix}/{timestamp}.png"
        
        # Upload to S3
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        self.s3.upload_fileobj(
            buffer, 
            self.bucket, 
            key,
            ExtraArgs={'ContentType': 'image/png'}
        )
        
        return f"https://{self.bucket}.s3.amazonaws.com/{key}"
    
    def save_interpolation(self, frames: list, prefix: str = "chain") -> str:
        """Save interpolation frames as GIF."""
        images = []
        for frame in frames:
            img_array = (frame.squeeze().cpu().numpy() * 255).astype('uint8')
            images.append(Image.fromarray(img_array, mode='L'))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key = f"{prefix}/{timestamp}.gif"
        
        buffer = io.BytesIO()
        images[0].save(
            buffer,
            format='GIF',
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        buffer.seek(0)
        
        self.s3.upload_fileobj(
            buffer,
            self.bucket,
            key,
            ExtraArgs={'ContentType': 'image/gif'}
        )
        
        return f"https://{self.bucket}.s3.amazonaws.com/{key}"
```

### Step 2.3: Database Service

```python
# src/database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class GeneratedImage(Base):
    __tablename__ = 'generated_images'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    prompt = Column(String)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    temperature = Column(Float, default=1.0)
    image_url = Column(String, nullable=False)
    user_session = Column(String(100))
    is_in_chain = Column(Boolean, default=False)


class ChainLink(Base):
    __tablename__ = 'chain_links'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    image_a_id = Column(Integer, ForeignKey('generated_images.id'))
    image_b_id = Column(Integer, ForeignKey('generated_images.id'))
    interpolation_url = Column(String, nullable=False)
    position = Column(Integer, nullable=False)
    
    image_a = relationship("GeneratedImage", foreign_keys=[image_a_id])
    image_b = relationship("GeneratedImage", foreign_keys=[image_b_id])


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def add_image(self, **kwargs) -> GeneratedImage:
        session = self.Session()
        image = GeneratedImage(**kwargs)
        session.add(image)
        session.commit()
        session.refresh(image)
        return image
    
    def get_latest_images(self, limit: int = 50) -> list[GeneratedImage]:
        session = self.Session()
        return session.query(GeneratedImage)\
            .order_by(GeneratedImage.created_at.desc())\
            .limit(limit)\
            .all()
    
    def get_chain(self) -> list[ChainLink]:
        session = self.Session()
        return session.query(ChainLink)\
            .order_by(ChainLink.position)\
            .all()
    
    def add_to_chain(self, image_a_id: int, image_b_id: int, interp_url: str) -> ChainLink:
        session = self.Session()
        
        # Get next position
        last = session.query(ChainLink).order_by(ChainLink.position.desc()).first()
        next_pos = (last.position + 1) if last else 0
        
        link = ChainLink(
            image_a_id=image_a_id,
            image_b_id=image_b_id,
            interpolation_url=interp_url,
            position=next_pos
        )
        session.add(link)
        
        # Mark images as in chain
        session.query(GeneratedImage).filter(
            GeneratedImage.id.in_([image_a_id, image_b_id])
        ).update({GeneratedImage.is_in_chain: True}, synchronize_session=False)
        
        session.commit()
        return link
```

---

## Phase 3: The Growing Chain (Background Worker)

### Concept

Every new generated image triggers:
1. Save image to storage + database
2. Find the "end" of the current chain
3. Create interpolation from chain end → new image
4. Add interpolation to chain
5. Broadcast update to all connected clients

### Step 3.1: Background Worker with Redis Queue

```python
# src/worker.py

import redis
import json
import torch
from models import FashionVAE
from storage import ImageStorage
from database import Database

class ChainWorker:
    """Background worker that builds the interpolation chain."""
    
    def __init__(self, redis_url: str, db_url: str, storage_bucket: str):
        self.redis = redis.from_url(redis_url)
        self.db = Database(db_url)
        self.storage = ImageStorage(storage_bucket)
        
        # Load VAE
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.vae = FashionVAE(latent_dim=32)
        self.vae.load_state_dict(torch.load("weights/vae.pth", map_location=self.device))
        self.vae.to(self.device).eval()
    
    def process_queue(self):
        """Main worker loop."""
        while True:
            # Blocking pop from queue
            _, message = self.redis.brpop("chain_queue")
            job = json.loads(message)
            
            try:
                self.add_to_chain(job["image_id"])
            except Exception as e:
                print(f"Error processing job: {e}")
    
    def add_to_chain(self, new_image_id: int):
        """Add a new image to the chain."""
        session = self.db.Session()
        
        # Get the new image
        new_image = session.query(GeneratedImage).get(new_image_id)
        if not new_image:
            return
        
        # Get the last image in chain
        last_link = session.query(ChainLink)\
            .order_by(ChainLink.position.desc())\
            .first()
        
        if last_link:
            last_image = last_link.image_b
        else:
            # First image - just mark it
            new_image.is_in_chain = True
            session.commit()
            self.broadcast_update()
            return
        
        # Generate interpolation
        with torch.no_grad():
            frames = self.vae.interpolate_smooth(
                last_image.class_id,
                new_image.class_id,
                steps=8,
                device=self.device
            )
        
        # Save interpolation
        interp_url = self.storage.save_interpolation(frames)
        
        # Add to chain
        self.db.add_to_chain(last_image.id, new_image.id, interp_url)
        
        # Broadcast to clients
        self.broadcast_update()
    
    def broadcast_update(self):
        """Notify all clients of chain update."""
        self.redis.publish("chain_updates", json.dumps({
            "type": "chain_updated",
            "timestamp": datetime.utcnow().isoformat()
        }))
```

### Step 3.2: Queue New Images

```python
# Add to api.py

import redis
import json

redis_client = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))

def queue_for_chain(image_id: int):
    """Add image to chain processing queue."""
    redis_client.lpush("chain_queue", json.dumps({"image_id": image_id}))


# Update generate endpoint
@app.post("/generate/prompt", response_model=PromptResponse)
async def generate_from_prompt(request: PromptRequest):
    # ... existing generation code ...
    
    # Save to database and queue for chain
    for i, img_tensor in enumerate(samples):
        url = storage.save_image(img_tensor)
        db_image = db.add_image(
            prompt=request.prompt,
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            temperature=params.get("temperature", 1.0),
            image_url=url,
        )
        
        # Add first image of each generation to chain
        if i == 0:
            queue_for_chain(db_image.id)
    
    # ... return response ...
```

---

## Phase 4: Real-Time Updates (WebSockets)

### Step 4.1: WebSocket Endpoint

```python
# src/websocket.py

from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import redis.asyncio as aioredis

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# Add to api.py
@app.websocket("/ws/chain")
async def websocket_chain(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Subscribe to Redis pubsub
    redis = aioredis.from_url(os.environ.get("REDIS_URL"))
    pubsub = redis.pubsub()
    await pubsub.subscribe("chain_updates")
    
    try:
        # Send current chain state
        chain = db.get_chain()
        await websocket.send_json({
            "type": "initial_state",
            "chain": [
                {
                    "position": link.position,
                    "interpolation_url": link.interpolation_url,
                }
                for link in chain
            ]
        })
        
        # Listen for updates
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await websocket.send_json(data)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## Phase 5: Frontend (Next.js)

### Step 5.1: Project Structure

```
fashion-web/
├── app/
│   ├── page.tsx              # Main page
│   ├── layout.tsx            # Root layout
│   └── api/                   # API routes (optional proxy)
├── components/
│   ├── PromptInput.tsx       # Text input with autocomplete
│   ├── GeneratedGallery.tsx  # Display generated images
│   ├── ChainBackground.tsx   # Animated chain background
│   └── InterpolationViewer.tsx
├── hooks/
│   ├── useChainWebSocket.ts  # WebSocket connection
│   └── useGenerate.ts        # Generation API calls
├── lib/
│   └── api.ts                # API client
└── public/
```

### Step 5.2: Chain Background Component

```tsx
// components/ChainBackground.tsx

'use client';

import { useEffect, useState, useRef } from 'react';
import { useChainWebSocket } from '@/hooks/useChainWebSocket';

interface ChainLink {
  position: number;
  interpolation_url: string;
}

export function ChainBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { chain, isConnected } = useChainWebSocket();
  const [loadedImages, setLoadedImages] = useState<HTMLImageElement[]>([]);
  
  // Load images as chain grows
  useEffect(() => {
    const loadImages = async () => {
      const images = await Promise.all(
        chain.map(link => {
          return new Promise<HTMLImageElement>((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = link.interpolation_url;
          });
        })
      );
      setLoadedImages(images);
    };
    
    loadImages();
  }, [chain]);
  
  // Animate the chain
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    let offset = 0;
    const imageSize = 56;  // 28 * 2 for visibility
    const speed = 0.5;
    
    const animate = () => {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw chain as scrolling strip
      loadedImages.forEach((img, i) => {
        const x = (i * imageSize - offset) % (canvas.width + imageSize * loadedImages.length);
        const y = canvas.height / 2 - imageSize / 2;
        
        ctx.globalAlpha = 0.3;
        ctx.drawImage(img, x, y, imageSize, imageSize);
      });
      
      offset += speed;
      requestAnimationFrame(animate);
    };
    
    animate();
  }, [loadedImages]);
  
  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10"
      width={typeof window !== 'undefined' ? window.innerWidth : 1920}
      height={typeof window !== 'undefined' ? window.innerHeight : 1080}
    />
  );
}
```

### Step 5.3: Prompt Input Component

```tsx
// components/PromptInput.tsx

'use client';

import { useState, useEffect } from 'react';
import { useGenerate } from '@/hooks/useGenerate';

export function PromptInput() {
  const [prompt, setPrompt] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const { generate, isLoading, result } = useGenerate();
  
  // Fetch autocomplete suggestions
  useEffect(() => {
    if (prompt.length < 2) {
      setSuggestions([]);
      return;
    }
    
    const fetchSuggestions = async () => {
      const res = await fetch(`/api/suggest?q=${encodeURIComponent(prompt)}`);
      const data = await res.json();
      setSuggestions(data.suggestions);
    };
    
    const timeout = setTimeout(fetchSuggestions, 200);
    return () => clearTimeout(timeout);
  }, [prompt]);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) {
      generate(prompt);
    }
  };
  
  return (
    <div className="w-full max-w-xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe a fashion item... (e.g., 'casual sneaker')"
          className="w-full px-6 py-4 text-lg rounded-full border-2 border-gray-300 
                     focus:border-pink-500 focus:outline-none"
          disabled={isLoading}
        />
        
        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className="absolute right-2 top-2 px-6 py-2 bg-pink-500 text-white 
                     rounded-full hover:bg-pink-600 disabled:opacity-50"
        >
          {isLoading ? 'Generating...' : 'Generate'}
        </button>
        
        {/* Autocomplete dropdown */}
        {suggestions.length > 0 && (
          <ul className="absolute top-full left-0 right-0 bg-white border rounded-lg mt-1 shadow-lg">
            {suggestions.map((suggestion) => (
              <li
                key={suggestion}
                onClick={() => setPrompt(suggestion)}
                className="px-4 py-2 hover:bg-gray-100 cursor-pointer"
              >
                {suggestion}
              </li>
            ))}
          </ul>
        )}
      </form>
      
      {/* Results */}
      {result && (
        <div className="mt-8">
          <p className="text-center text-gray-600 mb-4">
            Detected: <strong>{result.detected_class}</strong>
            {result.confidence === 'random' && ' (random - no match found)'}
          </p>
          
          <div className="grid grid-cols-4 gap-4">
            {result.images.map((img, i) => (
              <img
                key={i}
                src={`data:image/png;base64,${img}`}
                alt={`Generated ${i + 1}`}
                className="w-full aspect-square rounded-lg shadow-md"
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## Phase 6: Deployment

### Option A: Vercel + Railway + Cloudflare R2

```
Frontend (Next.js) → Vercel (free tier)
Backend (FastAPI)  → Railway ($5/mo)
Database           → Railway PostgreSQL
Redis              → Railway Redis
Storage            → Cloudflare R2 (10GB free)
```

### Option B: Single VPS (DigitalOcean/Hetzner)

```yaml
# docker-compose.yml

version: '3.8'

services:
  frontend:
    build: ./fashion-web
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8080
  
  api:
    build: ./fashion_item_generator
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fashion
      - REDIS_URL=redis://redis:6379
      - S3_BUCKET=fashion-generated
    depends_on:
      - db
      - redis
  
  worker:
    build: ./fashion_item_generator
    command: python -m src.worker
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fashion
      - REDIS_URL=redis://redis:6379
      - S3_BUCKET=fashion-generated
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=fashion
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
  
  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

---

## Implementation Order

### Week 1: Prompt-to-Generation
- [ ] Create `prompt_parser.py`
- [ ] Add `/generate/prompt` endpoint
- [ ] Test locally with Streamlit

### Week 2: Database + Storage
- [ ] Set up PostgreSQL (local Docker)
- [ ] Set up Cloudflare R2 or MinIO
- [ ] Implement `database.py` and `storage.py`
- [ ] Update API to save all generations

### Week 3: Chain System
- [ ] Set up Redis
- [ ] Implement `worker.py`
- [ ] Add WebSocket endpoint
- [ ] Test chain growth locally

### Week 4: Frontend
- [ ] Create Next.js project
- [ ] Build PromptInput component
- [ ] Build ChainBackground component
- [ ] Connect WebSocket

### Week 5: Deploy & Polish
- [ ] Deploy backend to Railway
- [ ] Deploy frontend to Vercel
- [ ] Set up production database
- [ ] Monitor and fix bugs

---

## Future Enhancements

### Phase 7: CLIP Integration (Better Prompts)
- Fine-tune CLIP on fashion vocabulary
- Map CLIP embeddings to VAE latent space
- Handle complex prompts like "vintage 80s sneaker with neon colors"

### Phase 8: User Accounts
- Save generation history
- "Like" images to influence chain priority
- User-generated interpolation paths

### Phase 9: Community Features
- Global leaderboard of most-liked generations
- Daily/weekly themes
- Collaborative chain branches

---

## Cost Estimate (Monthly)

| Service | Free Tier | Paid |
|---------|-----------|------|
| Vercel | 100GB bandwidth | $20 |
| Railway | $5 credit | $10-20 |
| Cloudflare R2 | 10GB | $0.015/GB |
| Domain | - | $12/year |

**Minimum viable: ~$10-15/month**
