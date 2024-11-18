import tensorflow as tf

def build_model(input_shape, num_categories):
    """
    Build CNN model
    input_shape: (height, width, channels)
    num_categories: number of fruit to classify
    returns build model
    """

    data_dir = "../data/raw"

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(9, activation='softmax')  # 9 output classes for fruit categories
    ])
    
    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


    
