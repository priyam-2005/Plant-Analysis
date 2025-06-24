import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from textblob import TextBlob
import streamlit as st

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    r"C:\Users\HP\Downloads\archive (5)\PlantVillage",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    r"C:\Users\HP\Downloads\archive (5)\PlantVillage",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the model
model.save("plant_disease_model.keras")

# --- Constants ---
MODEL_PATH = 'plant_disease_model.keras'
TEMP_IMAGE_PATH = "temp_img.jpg"
LABELS = ['Healthy', 'Early_Blight', 'Late_Blight']  # Update this based on your model

# --- Load Model Safely ---
@st.cache_resource(show_spinner=False)
def load_cnn_model(path):
    if not os.path.exists(path):
        st.error("❌ Model file not found. Please upload `plant_disease_model.h5`.")
        return None
    return load_model(path)

model = load_cnn_model(MODEL_PATH)

# --- Image Preprocessing ---
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

# --- Disease Prediction ---
def predict_disease(img_path):
    img = preprocess_image(img_path)
    if img is None:
        return "Error"
    prediction = model.predict(img)[0]
    return LABELS[np.argmax(prediction)]

# --- NLP Sentiment Analysis ---
def analyze_farmer_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if any(word in text.lower() for word in ['yellow', 'leaf', 'spot']):
        issue = "🟡 Possible fungal or nutrient issue."
    elif any(word in text.lower() for word in ['dry', 'wilting']):
        issue = "💧 Check for water stress or soil quality."
    elif polarity < -0.2:
        issue = "⚠️ Farmer is concerned. Immediate attention recommended."
    else:
        issue = "✅ General monitoring suggested."

    return polarity, issue

# --- Streamlit UI ---
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease & Farmer Query Analyzer")
st.markdown("Upload a plant image and describe the issue for automated analysis.")

img_file = st.file_uploader("📷 Upload Plant Image", type=["jpg", "png", "jpeg"])
farmer_input = st.text_area("📝 Describe the issue (in any language)")

if img_file and farmer_input and model:
    # Save uploaded image
    with open(TEMP_IMAGE_PATH, "wb") as f:
        f.write(img_file.read())

    # Disease Detection
    st.subheader("🧪 Analyzing Image...")
    disease_result = predict_disease(TEMP_IMAGE_PATH)

    # Text Sentiment Analysis
    st.subheader("📖 Analyzing Farmer's Description...")
    sentiment_score, suggestion = analyze_farmer_text(farmer_input)

    # Show Results
    st.image(TEMP_IMAGE_PATH, caption="Uploaded Image", use_column_width=True)
    st.success(f"🦠 **Detected Disease**: {disease_result}")
    st.info(f"🧠 **Sentiment Score**: {sentiment_score:.2f}")
    st.warning(suggestion)

    # Clean up (optional)
    os.remove(TEMP_IMAGE_PATH)
elif not model:
    st.stop()
# Display a message if no image or text is provide