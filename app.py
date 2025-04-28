import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile

# Load model
model = tf.keras.models.load_model(r"C:\Users\skgau\OneDrive\Desktop\Project_opencv\skin_disease_model_updated.h5")

# Class labels
class_names = [
    'Cellulitis', 'Impetigo', 'Athlete-foot', 'Nail-fungus',
    'Ringworm', 'Cutaneous-larva-migrans', 'Chickenpox', 'Shingles'
]

# Confidence threshold (tune this higher for stricter detection)
CONFIDENCE_THRESHOLD = 90

# UI
st.title("🩺 Skin Disease Detector")
st.write("Upload an image OR use your webcam to get predictions.")

# Upload or camera input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or take a photo")

# Prediction function
def preprocess_and_predict(img):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype("float32") / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        prediction = model.predict(img_expanded)
        confidence = np.max(prediction) * 100
        predicted_class = class_names[np.argmax(prediction)]

        # Debug: Show raw scores
        # st.write("🔍 Raw prediction scores:", prediction.tolist())

        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "No disease detected"

        return predicted_class, confidence, img_resized
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0.0, img

# Handle file upload
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    img = cv2.imread(tfile.name)
    if img is None:
        st.error("❌ Failed to load the image. Try another file.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predicted_class, confidence, processed_img = preprocess_and_predict(img)

        st.image(processed_img, caption="Uploaded Image", use_column_width=True)

        if predicted_class == "No disease detected":
            st.markdown("### ⚠️ No visible skin disease detected.")
            st.markdown(f"**🧠 Confidence in best match:** `{confidence:.2f}%` (Below threshold)")
        else:
            st.markdown(f"### ✅ Prediction: `{predicted_class}`")
            st.markdown(f"**🧠 Confidence:** `{confidence:.2f}%`")

# Handle webcam input
elif camera_image is not None:
    img_bytes = camera_image.getvalue()
    np_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Could not process webcam image. Please try again.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predicted_class, confidence, processed_img = preprocess_and_predict(img)

        st.image(processed_img, caption="Webcam Image", use_column_width=True)

        if predicted_class == "No disease detected":
            st.markdown("### ⚠️ No visible skin disease detected.")
            st.markdown(f"**🧠 Confidence in best match:** `{confidence:.2f}%` (Below threshold)")
        else:
            st.markdown(f"### ✅ Prediction: `{predicted_class}`")
            st.markdown(f"**🧠 Confidence:** `{confidence:.2f}%`")
