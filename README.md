# Skin Cancer Detection Web App

This is a **Streamlit-based web application** that allows users to upload an image of a skin lesion and receive a prediction of its class using a trained deep learning model.

## 🧠 Model Information

The model used is a pre-trained Keras model (`skin_cancer_model.h5`) capable of classifying images into one of the following skin lesion categories:

- Melanocytic nevi  
- Melanoma  
- Benign keratosis-like lesions  
- Basal cell carcinoma  
- Actinic keratoses  
- Vascular lesions  
- Dermatofibroma  

## 📦 Requirements

To run this app, you'll need the following Python packages:

- `streamlit`
- `keras` (with TensorFlow backend)
- `Pillow`
- `numpy`

Install the dependencies using:

```bash
pip install streamlit keras Pillow numpy
