# Skin Disease Detector

This project is a **skin disease classification web application** built with **Streamlit** and **TensorFlow**.  
It uses a pre-trained deep learning model to detect eight types of common skin diseases from an image captured via webcam or uploaded manually.

---

## 🚀 Features

- Predicts skin diseases from images using a trained CNN model.
- Supports both image uploads and real-time webcam capture.
- Displays the prediction along with the model's confidence percentage.
- Alerts if no visible disease is detected (based on a confidence threshold).
- Simple and intuitive UI built with Streamlit.

---

## 📂 Project Structure

```bash
.
├── app.py                          # Main Streamlit application
├── skin_disease_model_with_opencv.ipynb # Notebook used to build/train/test the model
├── skin_disease_model_updated.h5   # (Required) Pre-trained model
└── README.md                       # Project documentation



 
