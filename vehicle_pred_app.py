import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Sidebar with app information
st.sidebar.title("About")
st.sidebar.info(
    """
    **Vehicle Type Prediction App** \n
    Upload a vehicle image, and the model will predict the vehicle type. \n
    Suported vehicle types:
    - Auto Rickshaws 
    - Bikes 
    - Cars 
    - Motorcycles
    - Planes 
    - Ships 
    - Trains 
    """
)

def create_model():
    base_model = tf.keras.applications.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
    model = Sequential()
    model.add(base_model)
    for layer in model.layers:
        layer.trainable = False  # Freeze the layers of mobilenet
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the weights
model_path = "efficientnet1.h5"
try:
    model.load_weights(model_path)
except Exception as e:
    st.error(f"Error loading weights: {e}")

# Main app interface
st.title('Vehicle Type Prediction App')
st.write("Upload a vehicle image, and the model will predict the type of vehicle.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    # Display the image in the first column
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.convert("RGB")
    img = img.resize((240, 240))  # Resize to match MobileNet input
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Use EfficientNetB1 preprocessing

    # Vehicle classes
    classes = ['Auto Rickshaw', 'Bike/Bicycle', 'Car', 'Motorcycle', 'Plane', 'Ship', 'Train']

    # Predict the vehicle type
    if st.button('Predict'):
        try:
            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]
            confidence_scores = prediction[0] * 100  # Convert to percentage

            # Display prediction results
            with col2:
                st.markdown(f'### Predicted Vehicle Type: **{predicted_class}**')
                st.markdown("### Confidence Scores:")
                for i, score in enumerate(confidence_scores):
                    st.markdown(f"**{classes[i]}:** {score:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


