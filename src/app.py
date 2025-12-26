#!/usr/bin/env python3
"""Streamlit interactive demo for Fashion-MNIST models."""

import io
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import streamlit as st

from models import FashionCNN, FashionVAE

# Get project root - resolve to absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Class names
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


@st.cache_resource
def load_models():
    """Load models (cached)."""
    device = get_device()
    
    # Load CNN
    cnn = FashionCNN()
    cnn_path = WEIGHTS_DIR / "cnn.pth"
    try:
        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        cnn.to(device).eval()
    except Exception as e:
        print(f"Failed to load CNN from {cnn_path}: {e}")
        cnn = None
    
    # Load VAE
    vae = FashionVAE(latent_dim=32)
    vae_path = WEIGHTS_DIR / "vae.pth"
    try:
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        vae.to(device).eval()
    except Exception as e:
        print(f"Failed to load VAE from {vae_path}: {e}")
        vae = None
    
    return cnn, vae, device


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess uploaded image for CNN."""
    image = image.convert("L").resize((28, 28))
    tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


def main():
    st.set_page_config(
        page_title="Fashion-MNIST Demo",
        page_icon="👕",
        layout="wide",
    )
    
    st.title("👕 Fashion-MNIST Demo")
    
    # Load models
    cnn, vae, device = load_models()
    
    # Sidebar status
    with st.sidebar:
        st.header("Model Status")
        st.write(f"**Device:** {device}")
        st.write(f"**CNN:** {'✅ Loaded' if cnn else '❌ Not found'}")
        st.write(f"**VAE:** {'✅ Loaded' if vae else '❌ Not found'}")
        
        st.divider()
        st.header("Classes")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{i}: {name}")
    
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs(["📤 Classify", "🎨 Generate", "🔄 Interpolate"])
    
    # Tab 1: Classification
    with tab1:
        st.markdown("Upload an image to classify and generate similar variations.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            uploaded = st.file_uploader(
                "Upload a fashion item (28x28 grayscale works best)",
                type=["png", "jpg", "jpeg"],
            )
        
        with col2:
            if uploaded is not None:
                image = Image.open(uploaded)
                
                st.subheader("Uploaded Image")
                st.image(image, width=150)
                
                if cnn is not None:
                    tensor = preprocess_image(image).to(device)
                    with torch.no_grad():
                        logits = cnn(tensor)
                        probs = torch.softmax(logits, dim=1).squeeze()
                        pred_class = probs.argmax().item()
                        confidence = probs[pred_class].item()
                    
                    st.subheader("Classification Result")
                    st.success(f"**{CLASS_NAMES[pred_class]}** ({confidence:.1%} confidence)")
                    
                    st.bar_chart({CLASS_NAMES[i]: probs[i].item() for i in range(10)})
                    
                    if vae is not None:
                        st.subheader(f"Generated Variations ({CLASS_NAMES[pred_class]})")
                        with torch.no_grad():
                            samples = vae.generate_class(pred_class, 4, device)
                        
                        cols = st.columns(4)
                        for i, col in enumerate(cols):
                            img = (samples[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                            col.image(img, width=100, caption=f"Sample {i+1}")
                else:
                    st.warning("CNN not loaded - cannot classify")
            else:
                st.info("👈 Upload an image to classify")
    
    # Tab 2: Generation
    with tab2:
        st.markdown("Generate fashion items by class.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_class = st.selectbox("Select class:", CLASS_NAMES, key="gen_class")
            num_samples = st.slider("Number of samples:", 1, 8, 4)
            generate_btn = st.button("Generate", type="primary")
        
        with col2:
            if generate_btn:
                if vae is not None:
                    class_id = CLASS_NAMES.index(selected_class)
                    
                    st.subheader(f"Generated: {selected_class}")
                    with torch.no_grad():
                        samples = vae.generate_class(class_id, num_samples, device)
                    
                    cols = st.columns(min(num_samples, 4))
                    for i in range(num_samples):
                        col = cols[i % 4]
                        img = (samples[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                        col.image(img, width=100, caption=f"Sample {i+1}")
                else:
                    st.warning("VAE not loaded - cannot generate")
            else:
                st.info("👈 Select a class and click Generate")
    
    # Tab 3: Interpolation
    with tab3:
        st.markdown("Morph between two fashion classes in latent space.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            class1 = st.selectbox("From class:", CLASS_NAMES, index=0, key="interp_from")
            class2 = st.selectbox("To class:", CLASS_NAMES, index=3, key="interp_to")
            steps = st.slider("Interpolation steps:", 4, 16, 8)
            smooth = st.checkbox("Use smooth interpolation (slerp)", value=True)
            interp_btn = st.button("Interpolate", type="primary", key="interp_btn")
        
        with col2:
            if interp_btn:
                if vae is not None:
                    class1_id = CLASS_NAMES.index(class1)
                    class2_id = CLASS_NAMES.index(class2)
                    
                    st.subheader(f"{class1} → {class2}")
                    
                    with torch.no_grad():
                        if smooth:
                            frames = vae.interpolate_smooth(class1_id, class2_id, steps, device)
                        else:
                            frames = vae.interpolate(class1_id, class2_id, steps, device)
                    
                    # Display frames
                    cols = st.columns(min(steps, 8))
                    for i in range(steps):
                        col = cols[i % 8]
                        img = (frames[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                        col.image(img, width=80)
                    
                    # Create GIF
                    pil_frames = []
                    for i in range(steps):
                        img = (frames[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                        pil_frames.append(Image.fromarray(img, mode='L').resize((112, 112), Image.NEAREST))
                    
                    # Save GIF to bytes
                    gif_buffer = io.BytesIO()
                    pil_frames[0].save(
                        gif_buffer,
                        format='GIF',
                        save_all=True,
                        append_images=pil_frames[1:] + pil_frames[-2::-1],  # Forward + backward
                        duration=150,
                        loop=0
                    )
                    gif_buffer.seek(0)
                    
                    st.download_button(
                        "📥 Download GIF",
                        gif_buffer,
                        file_name=f"morph_{class1}_{class2}.gif",
                        mime="image/gif"
                    )
                else:
                    st.warning("VAE not loaded - cannot interpolate")
            else:
                st.info("👈 Select two classes and click Interpolate")


if __name__ == "__main__":
    main()
