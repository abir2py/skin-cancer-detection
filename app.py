import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import io

@st.cache_resource
def load_trained_model():
    return load_model("skin_cancer_model.h5")

model = load_trained_model()

class_labels = [
    'Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions',
    'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma'
]

x_train_mean = 0.5
x_train_std = 0.25

def preprocess_image(image: Image.Image):
    image = image.resize((100, 75)) 
    img_array = np.asarray(image) / 255.0  
    img_array = (img_array - x_train_mean) / x_train_std  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


st.title("Skin Cancer Detection")
st.write("Upload a skin lesion image and get the predicted class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        input_image = preprocess_image(image)
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {class_labels[predicted_class]} ({confidence:.2f}% confidence)")
