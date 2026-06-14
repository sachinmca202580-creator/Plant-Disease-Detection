import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="wide"
)


# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- CLASS LABELS ----------------
class_labels = {
    0: "Blight",
    1: "Common Rust",
    2: "Gray Leaf Spot",
    3: "Healthy"
}


# ---------------- DISEASE INFORMATION ----------------
disease_info = {
    "Blight": "Blight causes brown or dark patches on maize leaves. It can spread quickly and reduce crop yield if not controlled early.",
    "Common Rust": "Common Rust appears as reddish-orange or brown spots on leaves. It affects leaf health and reduces photosynthesis.",
    "Gray Leaf Spot": "Gray Leaf Spot produces gray or brown rectangular lesions on leaves. It can damage large leaf areas and weaken the plant.",
    "Healthy": "The leaf appears healthy. No visible disease symptoms are detected by the model."
}


# ---------------- PRECAUTION / TREATMENT ----------------
precautions = {
    "Blight": "Remove infected leaves, avoid overhead watering, maintain field hygiene, and use recommended fungicide if infection spreads.",
    "Common Rust": "Improve air circulation, avoid excessive moisture, remove infected leaves, and apply rust-control fungicide if required.",
    "Gray Leaf Spot": "Avoid excess moisture, maintain proper plant spacing, remove affected leaves, and use suitable fungicide treatment.",
    "Healthy": "No treatment is required. Continue regular watering, proper sunlight, and periodic monitoring."
}


# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(in_features=512, out_features=4)

    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


model = load_model()


# ---------------- IMAGE TRANSFORM ----------------
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
    all_probabilities = probabilities.cpu().numpy()[0] * 100

    return predicted_label, confidence_score, all_probabilities


# ---------------- UI ----------------
st.title("🌿 Plant Disease Detection Using AI")
st.markdown(
    """
    This application detects maize plant leaf disease using a **ResNet18 CNN model**.
    You can upload a leaf image or capture one using your camera.
    """
)

st.sidebar.header("📌 Project Details")
st.sidebar.write("**Project:** Plant Disease Detection Using AI")
st.sidebar.write("**Model:** ResNet18 CNN")
st.sidebar.write("**Framework:** PyTorch")
st.sidebar.write("**Frontend:** Streamlit")
st.sidebar.write("**Classes:** Blight, Common Rust, Gray Leaf Spot, Healthy")

input_method = st.radio(
    "Choose input method:",
    ["Upload Image", "Camera Scan"]
)

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a maize leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

else:
    camera_file = st.camera_input("Capture maize leaf image")

    if camera_file is not None:
        image = Image.open(camera_file)


if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selected Leaf Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        predicted_label, confidence_score, probabilities = predict_image(image)

        st.success(f"Predicted Disease: {predicted_label}")
        st.info(f"Confidence Score: {confidence_score:.2f}%")

        st.subheader("Disease Description")
        st.write(disease_info[predicted_label])

        st.subheader("Precaution / Treatment")
        st.write(precautions[predicted_label])

    st.subheader("Prediction Probability Graph")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(class_labels.values()), probabilities)
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Disease Classes")
    ax.set_title("Model Prediction Probability")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)

    st.pyplot(fig)

else:
    st.warning("Please upload or capture a maize leaf image to start detection.")