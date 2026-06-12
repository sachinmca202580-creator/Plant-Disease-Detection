import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL LOAD ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(in_features=512, out_features=4)
model = model.to(device)

model_path = "model.pth"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ---------------- CLASS LABELS ----------------
class_labels = {
    0: "Blight",
    1: "Common_Rust",
    2: "Gray_Leaf_Spot",
    3: "Healthy"
}

# ---------------- DISEASE DESCRIPTION ----------------
disease_info = {
    "Blight": "Blight causes brown or dark patches on leaves and can spread quickly if not controlled.",
    "Common_Rust": "Common Rust appears as reddish-orange spots on leaves and affects plant growth.",
    "Gray_Leaf_Spot": "Gray Leaf Spot causes gray or brown lesions on leaves and reduces photosynthesis.",
    "Healthy": "The plant leaf appears healthy with no visible disease symptoms."
}

# ---------------- TREATMENT SUGGESTION ----------------
treatment = {
    "Blight": "Remove infected leaves, avoid overhead watering, and apply recommended fungicide.",
    "Common_Rust": "Use resistant varieties, improve air circulation, and apply fungicide if needed.",
    "Gray_Leaf_Spot": "Avoid excess moisture, remove affected leaves, and use proper fungicide treatment.",
    "Healthy": "No treatment required. Continue regular monitoring and proper watering."
}

# ---------------- IMAGE PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_labels[predicted_class.item()]
    confidence_score = confidence.item() * 100

    return predicted_label, confidence_score, probabilities.cpu().numpy()[0]

# ---------------- UI ----------------
st.title("🌿 Plant Disease Detection Using AI")
st.markdown("Upload a plant leaf image and the AI model will predict the plant condition.")

st.sidebar.title("📌 Project Information")
st.sidebar.write("**Model:** ResNet18 CNN")
st.sidebar.write("**Framework:** PyTorch")
st.sidebar.write("**Interface:** Streamlit")
st.sidebar.write("**Classes:** 4")
st.sidebar.write("Blight, Common Rust, Gray Leaf Spot, Healthy")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        predicted_label, confidence_score, probabilities = predict_image(image)

        st.success(f"Predicted Disease: {predicted_label}")
        st.info(f"Confidence Score: {confidence_score:.2f}%")

        st.subheader("Disease Description")
        st.write(disease_info[predicted_label])

        st.subheader("Recommended Treatment")
        st.write(treatment[predicted_label])

    # ---------------- MATPLOTLIB GRAPH ----------------
    st.subheader("Prediction Probability Graph")

    fig, ax = plt.subplots()
    ax.bar(class_labels.values(), probabilities * 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Disease Prediction Probability")
    plt.xticks(rotation=20)

    st.pyplot(fig)

else:
    st.warning("Please upload a plant leaf image to get prediction.")