import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

def build_model(input_shape, num_categories=9):
    """
    Build CNN model
    input_shape: (height, width, channels)
    num_categories: number of fruit to classify
    returns build model
    """

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model to retain learned features

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(.5),
        Dense(num_categories, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for transfer learning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_feature_map(model, layer_name, preprocessed_image):
    """
    Returns a feature map of how the model sees a image in a specicific layer.
    Creates intermediate model of specific layer to output the feature map
    """
    intermediate = tf.keras.Model(
        inputs=model.get_layer("mobilenetv2_1.00_224").input,  # Input tensor from the base model
        outputs=model.get_layer("mobilenetv2_1.00_224").get_layer(layer_name).output
    )
    feature_map = intermediate.predict(preprocessed_image)
    return feature_map

def predict_image_confidence(model, processed_image, class_names):
    """
    Clasify a preprocessed image,
    
    """
    results = model.predict(processed_image)
    result_index = np.argmax(results)
    result_class = class_names[result_index]
    confidence_scores = {class_names[i]: float(results[0][i]) for i in range(len(class_names))}

    return result_class, confidence_scores
