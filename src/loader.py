import os
from PIL import Image
import numpy as np
import tensorflow as tf


def load_images_with_labels(data_dir, image_size=(224, 224)):
    """
    Load images from the data dir. Directory names are used as the label for its contents

    data_dir - directory of dataset
    image_size - (width, height) resize image
    returns -  (images, labels) as numpy array
    """

    image_paths = []
    labels = []

    # Iterate through subdirectory (category) in the data
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)

        if os.path.isdir(category_path):
            # clean the label ("apple fruit" => "apple")
            label = category.split()[0]

            # iterate through each image and save the path and label
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                image_paths.append(image_path)
                labels.append(label)

    images = []
    encoded_labels = []

    # Create label to index mapping - converts into numerical representation for model
    # ex: {'apple': 0, 'banana': 1, 'cherry': 2, 'chickoo': 3, 'grapes': 4, 'kiwi': 5, 'mango': 6, 'orange': 7, 'strawberry': 8}
    label_index_map = {label: index for index, label in enumerate(sorted(set(labels)))}

    # process images, resize,
    for image_path, label in zip(image_paths, labels):
        image = Image.open(image_path).convert("RGB")
        image = image.resize(image_size)
        image_array = np.array(image)
        images.append(image_array)
        encoded_labels.append(label_index_map[label])

    images = np.array(images)
    encoded_labels = np.array(encoded_labels)

    images = images / 255.0

    return images, encoded_labels, label_index_map

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Loads and preprocesses an image to match the model's expected input format.
    """
    img = Image.open(image_path).convert("RGB")  
    img = img.resize(img_size)  
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)