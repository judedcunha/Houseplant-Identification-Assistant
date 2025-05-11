# A simple Streamlit web app to classify plant species and show care tips

import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import os

# Configuration
NUM_CLASSES = 10
MODEL_PATH = 'plantnet_finetuned_resnet18.pth'
CLASS_NAMES = [
    'Daucus carota', 'Alliaria petiolata', 'Hypericum perforatum',
    'Centranthus ruber', 'Cirsium vulgare', 'Trifolium pratense',
    'Calendula officinalis', 'Lamium purpureum', 'Alcea rosea', 'Papaver rhoeas'
]

CARE_TIPS = {
    'Daucus carota': 'Full sun; light, well-drained soil; keep consistently moist. Sow in spring or fall.',
    'Alliaria petiolata': 'Tolerates shade; grows in disturbed soils; invasive—control spread.',
    'Hypericum perforatum': 'Full sun; dry to medium soil; drought tolerant once established.',
    'Centranthus ruber': 'Full sun; well-drained soil; cut back after flowering to promote rebloom.',
    'Cirsium vulgare': 'Full sun; thrives in poor soils; invasive—control if not desired.',
    'Trifolium pratense': 'Full sun to partial shade; moist, fertile soil; fixes nitrogen naturally.',
    'Calendula officinalis': 'Full sun; moderate watering; deadhead for continuous bloom.',
    'Lamium purpureum': 'Partial shade; moist, rich soil; often grows as a groundcover.',
    'Alcea rosea': 'Full sun; well-drained soil; stake tall varieties for support.',
    'Papaver rhoeas': 'Full sun; light soil; sow directly—does not transplant well.'
}

# Load model
def load_model():
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Inference function
def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    label = CLASS_NAMES[predicted.item()]
    tip = CARE_TIPS.get(label, "No care tip available.")
    return label, tip

# Streamlit UI
st.title("🌿 Plant Identifier and Care Advisor")

uploaded_file = st.file_uploader("Upload a plant image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Classifying...'):
        model = load_model()
        label, tip = predict(image, model)

    st.success(f"**Identified:** {label}")
    st.markdown(f"**Care Tip:** {tip}")
