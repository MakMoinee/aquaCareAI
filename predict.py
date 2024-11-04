import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Path to the saved model
model_path = 'result.h5'

# Load the trained model
model = load_model(model_path)

# Path to the input image for white spot detection
image_path = './train/path/to/input_image.jpg'

# Image size (same as the input size used during training)
img_size = (299, 299)

# Function to preprocess and predict
def detect_white_spot(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image as done in training
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    # Interpret the result
    if prediction[0] > 0.5:
        print("White spot disease detected")
    else:
        print("No white spot disease detected")

# Run the detection function
detect_white_spot(image_path)
