import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape, num_categories):
    """
    Build CNN model
    input_shape: (height, width, channels)
    num_categories: number of fruit to classify
    returns build model
    """

    data_dir = "../data/raw"

    data_gen = ImageDataGenerator(
        validation_split = .20, 
        rescale= 1.0/255,
        rotation_range = 40,
        width_shift_range = .2,
        height_shift_range = .2,
        shear_range = .2,
        zoom_range=.2,
        horizontal_flip = True,
        fill_mode="nearest",
    )

    train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
    )

    validation_generator = data_gen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )


    
