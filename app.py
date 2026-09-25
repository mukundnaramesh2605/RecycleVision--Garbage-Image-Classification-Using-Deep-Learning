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

# ---------- ui ----------
st.title("♻️ RecycleVision")
st.caption("Garbage image classifier · ResNet50 · 12 classes · 97.4% test accuracy")
st.link_button("🔗 Link to GitHub repository", url="https://github.com/mukundnaramesh2605/RecycleVision--Garbage-Image-Classification-Using-Deep-Learning")

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
        st.warning(f"No images found in '{SAMPLE_DIR}/'. Add some and redeploy.")
    else:
        labels = [os.path.relpath(s, SAMPLE_DIR) for s in samples]
        choice = st.selectbox("Pick a sample image:", labels)
        show_results(Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB"))