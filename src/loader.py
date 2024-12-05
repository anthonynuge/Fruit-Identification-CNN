import os
from PIL import Image
import numpy as np
import tensorflow as tf

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Loads and preprocesses an image to match the model's expected input format.
    """
    img = Image.open(image_path).convert("RGB")  
    img = img.resize(img_size)  
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)