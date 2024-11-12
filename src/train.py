import os
import tensorflow as tf

# Constants
DATA_DIR = './data/raw'
MODELS_DIR = './models'
BATCH_SIZE = 16
EPOCHS = 30
IMG_SIZE = (224, 224)

def get_next_model_version_path(models_dir):
    """
    Dynamically name models based on currently saved models
    """
    os.makedirs(models_dir, exist_ok=True) 

    # Get all existing model files with the format 'modelV#.keras'
    model_files = [f for f in os.listdir(models_dir) if f.startswith("modelV") and f.endswith(".keras")]
    
    versions = [int(f[6:-6]) for f in model_files if f[6:-6].isdigit()]
    next_version = max(versions, default=0) + 1  
    
    # Construct the next model path
    return os.path.join(models_dir, f"modelV{next_version}.keras")

def create_data_generators(data_dir, img_size, batch_size):
    """Settings for data generator to simulate additional images. Images are randomly rotated and shifted. 
    Handles spliting training and validations as well. 80/20
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def build_model(input_shape, num_classes):
    """Build and return a CNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs, batch_size):
    """Train the model with given data generators and save it with an incremental version name."""
    # Calculate steps per epoch. Adds remainder to ensure full epochs
    steps_per_epoch = (train_generator.samples // batch_size) + int(train_generator.samples % batch_size != 0)
    validation_steps = (validation_generator.samples // batch_size) + int(validation_generator.samples % batch_size != 0)
    
    # Define learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[reduce_lr]
    )

    # Generate the next model save path
    model_save_path = get_next_model_version_path(MODELS_DIR)

    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    return history

if __name__ == '__main__':
    # Create data generators
    train_generator, validation_generator = create_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    # Build the model
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=train_generator.num_classes)

    # Train and save the model with an incremental version name
    history = train_model(model, train_generator, validation_generator, EPOCHS, BATCH_SIZE)

