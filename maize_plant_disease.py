# Streamlit app
st.title("🌿 Plant Disease Detection Using CNN")
st.write("Upload a plant leaf image to detect disease and get precautions.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Loading animation
    with st.spinner("🔍 Analyzing Leaf Image..."):
        result = predict_image(image)

    # Prediction Result
    st.success(f"✅ Predicted Disease: {result}")

    # Precautions Section
    st.subheader("🌱 Recommended Precautions")

    if result == "Healthy":
        st.info("""
        ✅ The leaf appears healthy.

        **Precautions:**
        - Provide proper water and sunlight.
        - Monitor leaves regularly.
        - Maintain soil fertility.
        - Avoid overwatering.
        - Keep the field clean and weed-free.
        """)

    elif result == "Common_Rust":
        st.warning("""
        ⚠️ Common Rust Detected

        **Precautions:**
        - Remove infected leaves immediately.
        - Avoid watering directly on leaves.
        - Improve air circulation around plants.
        - Apply recommended fungicide.
        - Monitor nearby plants for infection.
        """)

    elif result == "Blight":
        st.warning("""
        ⚠️ Blight Detected

        **Precautions:**
        - Remove infected plant parts.
        - Use disease-free seeds.
        - Avoid overhead irrigation.
        - Apply suitable fungicide.
        - Rotate crops to reduce disease spread.
        """)

    elif result == "Gray_Leaf_Spot":
        st.warning("""
        ⚠️ Gray Leaf Spot Detected

        **Precautions:**
        - Remove crop residue after harvest.
        - Use resistant maize varieties.
        - Maintain proper plant spacing.
        - Avoid excessive nitrogen fertilizer.
        - Apply fungicide if disease becomes severe.
        """)

    # Footer
    st.markdown("---")
    st.caption("Developed using CNN, PyTorch and Streamlit")