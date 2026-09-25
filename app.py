import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import models, transforms

# ---------- config ----------
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
MODEL_PATH = "models/resnet50.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")

# ---------- model (loaded once, cached) ----------
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# same preprocessing as training (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def predict(image):
    x = transform(image).unsqueeze(0)              # add batch dimension
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return probs

# ---------- ui ----------
st.title("♻️ RecycleVision")
st.caption("Garbage image classifier · ResNet50 · 12 classes · 97.4% test accuracy")

file = st.file_uploader("Upload a garbage image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    probs = predict(image)
    top = int(probs.argmax())

    st.subheader(f"Prediction: {CLASS_NAMES[top]}")
    st.write(f"Confidence: {probs[top]*100:.1f}%")

    st.write("**All classes:**")
    for i in probs.argsort()[::-1]:
        st.write(f"{CLASS_NAMES[i]} — {probs[i]*100:.1f}%")
        st.progress(float(probs[i]))
else:
    st.info("Upload an image to classify it.")