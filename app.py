import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import models, transforms

CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
MODEL_PATH = "models/resnet50.pt"
SAMPLE_DIR = "sample_images"
REPO_URL = "https://github.com/mukundnaramesh2605/RecycleVision--Garbage-Image-Classification-Using-Deep-Learning"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")

@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def predict(image):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return probs

def show_results(image):
    st.image(image, caption="Selected image", use_container_width=True)
    probs = predict(image)
    top = int(probs.argmax())
    st.subheader(f"Prediction: {CLASS_NAMES[top]}")
    st.write(f"Confidence: {probs[top]*100:.1f}%")
    st.write("**All classes:**")
    for i in probs.argsort()[::-1]:
        st.write(f"{CLASS_NAMES[i]} — {probs[i]*100:.1f}%")
        st.progress(float(probs[i]))

# ---------- header ----------
st.title("♻️ RecycleVision")
st.caption("Garbage image classifier · ResNet50 · 12 classes")

tab_about, tab_classify = st.tabs(["🏠 About", "🔍 Classify"])

# ---------- About tab ----------
with tab_about:
    st.header("About RecycleVision")
    st.write(
        "RecycleVision classifies a photo of a waste item into one of 12 categories to help "
        "automate recycling. It is built with transfer learning in PyTorch and serves a "
        "fine-tuned ResNet50 that reaches **97.4% accuracy** on the held-out test set."
    )

    st.subheader("The 12 categories")
    cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        cols[i % 3].write(f"- {name}")

    st.subheader("How it works")
    st.write(
        "- Three models were compared: a CNN from scratch, MobileNetV2, and ResNet50.\n"
        "- ResNet50 scored highest (97.4% accuracy, 0.968 macro-F1) and is the deployed model.\n"
        "- Your image is resized to 224×224 and normalized the same way as during training, "
        "then the model returns a probability for each of the 12 classes."
    )

    st.subheader("How to use it")
    st.write(
        "Open the **🔍 Classify** tab, then either upload your own image or pick a bundled "
        "sample. The app shows the predicted category, its confidence, and the full ranking."
    )

    st.link_button("🔗 View project on GitHub", REPO_URL)

# ---------- Classify tab ----------
with tab_classify:
    st.header("Classify an image")
    mode = st.radio("Choose input:", ["Upload an image", "Use a sample image"])

    if mode == "Upload an image":
        file = st.file_uploader("Upload a garbage image", type=["jpg", "jpeg", "png"])
        if file:
            show_results(Image.open(file).convert("RGB"))
        else:
            st.info("Upload an image to classify it.")
    else:
        samples = sorted(glob.glob(os.path.join(SAMPLE_DIR, "**", "*.*"), recursive=True))
        samples = [s for s in samples if s.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not samples:
            st.warning(f"No images found in '{SAMPLE_DIR}/'.")
        else:
            labels = [os.path.relpath(s, SAMPLE_DIR) for s in samples]
            choice = st.selectbox("Pick a sample image:", labels)
            show_results(Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB"))