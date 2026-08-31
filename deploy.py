import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


# Cache the model so it loads once per session instead of on every button
# click / page switch. Streamlit reruns this whole script top-to-bottom on
# every interaction, so without this decorator load_model() fired every time
# the user clicked "Predict".
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Model.h5')


model = load_model()

# The model's single sigmoid output is trained as P(REAL), because Keras
# assigns labels by alphabetical folder order (FAKE, REAL) -> (0, 1), and the
# training notebook confirms this via train_ds.class_names before training.
CLASS_NAMES = ["AI generated Image", "REAL Image"]


def preprocess_image(image):
    image = image.convert("RGB")  # handle uploads that aren't already RGB (e.g. PNG with alpha)
    image = image.resize((32, 32))
    img_array = np.array(image)
    img_array = img_array / 255.0
    return img_array


def predict(image):
    processed_image = preprocess_image(image)
    batch = np.expand_dims(processed_image, axis=0)  # shape (1, 32, 32, 3)
    prob_real = model.predict(batch)[0][0]  # single sigmoid output = P(REAL)
    label = CLASS_NAMES[1] if prob_real >= 0.5 else CLASS_NAMES[0]
    confidence = prob_real if prob_real >= 0.5 else 1 - prob_real
    return label, confidence


def main():
    st.title('AI vs REAL Image Classifier')
    pages = ['Home', 'Predictor']
    selected_page = st.sidebar.radio('Select a page', pages)
    if selected_page == 'Home':
        st.header('Welcome to the AI Image Classifier!')
        st.write('This app allows you to classify images as real or AI generated.')
    elif selected_page == 'Predictor':
        st.header('Image Predictor')
        uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write('')
            if st.button('Predict'):
                label, confidence = predict(image)
                st.write(f'Prediction: {label}')
                st.write(f'Confidence: {confidence:.2%}')


if __name__ == '__main__':
    main()
