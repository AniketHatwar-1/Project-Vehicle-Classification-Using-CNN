"""Simple web app for vehicle classification."""
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import VehicleCNN

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT/"models"/"vehicle_cnn.pth"
st.set_page_config(page_title="Vehicle Classifier", page_icon="🚗")
st.title("Vehicle Type Classifier")
st.write("Upload a vehicle photo. The CNN was trained from scratch on a small dataset.")
st.caption("Predictions are experimental; confidence is not a guarantee of correctness.")
if not MODEL_PATH.exists():
    st.warning("Model not found. Run train.bat first.")
    st.stop()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    size = int(ckpt["image_size"]); names = ckpt["class_names"]
    model = VehicleCNN(len(names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    prep = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(),
                               transforms.Normalize((0.5,)*3,(0.5,)*3)])
    return model, names, prep

model, names, prep = load_model()
file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp","bmp"])
if file:
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        with torch.no_grad():
            logits = model(prep(image).unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        order = sorted(range(len(names)), key=lambda i: probs[i], reverse=True)
        st.subheader(f"Prediction: {names[order[0]]}")
        st.metric("Model confidence", f"{probs[order[0]]:.1%}")
        st.write("Scores for all classes")
        for i in order:
            st.write(f"{names[i]} — {probs[i]:.1%}")
            st.progress(probs[i])
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
