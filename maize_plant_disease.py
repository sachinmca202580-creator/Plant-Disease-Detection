import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(in_features=512, out_features=4)
model = model.to(device)

model_path = "model.pth"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image):
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)

        class_labels = {
            0: "Blight",
            1: "Common_Rust",
            2: "Gray_Leaf_Spot",
            3: "Healthy"
        }

        return class_labels[predicted_class.item()]

    except Exception as e:
        return f"Error: {e}"


def show_precautions(result):
    st.subheader("🌱 Recommended Precautions")

    if result == "Healthy":
        st.info("""
        ✅ **Leaf Status: Healthy**

        - Provide proper water and sunlight.
        - Monitor leaves regularly.
        - Avoid overwatering.
        - Maintain soil fertility.
        - Keep the field clean and weed-free.
        """)

    elif result == "Common_Rust":
        st.warning("""
        ⚠️ **Disease Detected: Common Rust**

        - Remove infected leaves immediately.
        - Avoid watering directly on leaves.
        - Improve air circulation around plants.
        - Apply recommended fungicide.
        - Monitor nearby plants for infection.
        """)

    elif result == "Blight":
        st.warning("""
        ⚠️ **Disease Detected: Blight**

        - Remove infected plant parts.
        - Use disease-free seeds.
        - Avoid overhead irrigation.
        - Apply suitable fungicide.
        - Rotate crops to reduce disease spread.
        """)

    elif result == "Gray_Leaf_Spot":
        st.warning("""
        ⚠️ **Disease Detected: Gray Leaf Spot**

        - Remove crop residue after harvest.
        - Use resistant maize varieties.
        - Maintain proper plant spacing.
        - Avoid excessive nitrogen fertilizer.
        - Apply fungicide if disease becomes severe.
        """)


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection Using CNN")
st.write("Upload a plant leaf image to detect disease and get recommended precautions.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("🔍 Analyzing leaf image..."):
        result = predict_image(image)

    if result.startswith("Error"):
        st.error(result)
    else:
        st.success(f"✅ Predicted Disease: {result}")
        show_precautions(result)

    st.markdown("---")
    st.caption("Developed using CNN, PyTorch and Streamlit")