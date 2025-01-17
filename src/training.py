import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model

# Constants
BATCH_SIZE = 16
EPOCHS = 30
IMG_SIZE = (224, 224)
NUM_CATEGORIES = 9
AUTOTUNE = tf.data.experimental.AUTOTUNE

base_dir = os.path.dirname(os.path.abspath(__file__))
datas_dir = os.path.join(base_dir, "data")
DATA_DIR = os.path.join(datas_dir, "raw")
MODELS_DIR = os.path.join(base_dir, "models")

def data_split_augment(data_dir, img_size, batch_size, validation_split = .2):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=69,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        color_mode="rgb"
    )

    class_names = train_ds.class_names

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=69,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        color_mode="rgb"
    )

    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(.1),
        tf.keras.layers.RandomZoom(.1),
        tf.keras.layers.RandomContrast(.1),
    ])

    train_ds = train_ds.map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE )
    normalization_lay = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_lay(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_lay(x), y), num_parallel_calls=AUTOTUNE)

    # Cache and prefetch for performance optimization
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def save_model(model):
    # Takes a model and saves it in the model directory.
    # If there are already models increments the version number.
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = [m for m in os.listdir(MODELS_DIR) if m.startswith("modelV") and m.endswith(".keras")]
    versions = [int(f[6:-6]) for f in models if f[6:-6].isdigit()]
    next = max(versions, default=0) + 1
    save_dest = os.path.join(MODELS_DIR, f"modelV{next}.keras")
    model.save(save_dest)
    print("Model successful trained and saved")

def train_model():
    train_ds, val_ds, class_names = data_split_augment(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_categories=len(class_names))
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    save_model(model)

if __name__ == "__main__":
    train_model()