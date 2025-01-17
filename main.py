import os
import subprocess
import tensorflow as tf
from src.download_dataset import download_dataset
import tkinter as tk
from src.gui import FruitClassifierGui

# MODELS_DIR = './models'
# define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
raw_dir = os.path.join(data_dir, "raw")
train_script_path = os.path.join(base_dir, "src", "training.py")
MODELS_DIR = os.path.join(base_dir, "models")


def available_model(path=MODELS_DIR):
    """Check if there is a model trained and ready. First application boot model will be trained using data"""
    return any(os.path.isfile(os.path.join(path, m)) for m in os.listdir(path))

def load_newest_model(path=MODELS_DIR):
    """Load the most recent model"""
    models = [os.path.join(path, m) for m in os.listdir(path) if os.path.isfile(os.path.join(path, m))]

    if not models:
        raise FileNotFoundError("Model not found")

    newest = max(models, key=os.path.getmtime)
    model = tf.keras.models.load_model(newest)
    print(f"{newest} loaded")
    return model

def get_class_names():
    class_names = sorted([entry.name for entry in os.scandir(raw_dir) if entry.is_dir()])
    return class_names

def main():
    if available_model():
        print("Model Found. Launching gui")
        model = load_newest_model()
    else:
        print("No available model. Training a new one. This can take a awhile")
        subprocess.run(["python", train_script_path])
        model = load_newest_model()
    
    class_names = get_class_names()
    app = FruitClassifierGui(model = model, class_names = class_names)
    app.run()


if __name__ == "__main__":
    main()