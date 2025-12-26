#!/usr/bin/env python3
"""Streamlit interactive demo for Fashion-MNIST models."""

import torch
import numpy as np
from PIL import Image
import streamlit as st

from models import FashionCNN, FashionVAE

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
    try:
        cnn.load_state_dict(torch.load("weights/cnn.pth", map_location=device, weights_only=True))
        cnn.to(device).eval()
    except FileNotFoundError:
        cnn = None
    
    # Load VAE
    vae = FashionVAE(latent_dim=32)
    try:
        vae.load_state_dict(torch.load("weights/vae.pth", map_location=device, weights_only=True))
        vae.to(device).eval()
    except FileNotFoundError:
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
    st.markdown("Upload an image to classify and generate similar variations.")
    
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
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded = st.file_uploader(
            "Upload a fashion item (28x28 grayscale works best)",
            type=["png", "jpg", "jpeg"],
        )
        
        # Or select class to generate
        st.divider()
        st.header("🎨 Or Generate by Class")
        selected_class = st.selectbox("Select class:", CLASS_NAMES)
        num_samples = st.slider("Number of samples:", 1, 8, 4)
        generate_btn = st.button("Generate", type="primary")
    
    with col2:
        if uploaded is not None:
            # Process uploaded image
            image = Image.open(uploaded)
            
            st.subheader("Uploaded Image")
            st.image(image, width=150)
            
            if cnn is not None:
                # Classify
                tensor = preprocess_image(image).to(device)
                with torch.no_grad():
                    logits = cnn(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze()
                    pred_class = probs.argmax().item()
                    confidence = probs[pred_class].item()
                
                st.subheader("Classification Result")
                st.success(f"**{CLASS_NAMES[pred_class]}** ({confidence:.1%} confidence)")
                
                # Show probability bar chart
                st.bar_chart({CLASS_NAMES[i]: probs[i].item() for i in range(10)})
                
                # Generate variations
                if vae is not None:
                    st.subheader(f"Generated Variations ({CLASS_NAMES[pred_class]})")
                    with torch.no_grad():
                        samples = vae.generate(4, pred_class, device)
                    
                    cols = st.columns(4)
                    for i, col in enumerate(cols):
                        img = (samples[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                        col.image(img, width=100, caption=f"Sample {i+1}")
            else:
                st.warning("CNN not loaded - cannot classify")
        
        elif generate_btn:
            # Generate from selected class
            if vae is not None:
                class_id = CLASS_NAMES.index(selected_class)
                
                st.subheader(f"Generated: {selected_class}")
                with torch.no_grad():
                    samples = vae.generate(num_samples, class_id, device)
                
                cols = st.columns(min(num_samples, 4))
                for i in range(num_samples):
                    col = cols[i % 4]
                    img = (samples[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                    col.image(img, width=100, caption=f"Sample {i+1}")
            else:
                st.warning("VAE not loaded - cannot generate")
        
        else:
            st.info("👈 Upload an image or select a class to generate")


if __name__ == "__main__":
    main()
