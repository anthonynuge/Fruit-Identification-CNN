import os
from src.download_dataset import download_dataset
import tkinter as tk
from src.gui import FruitClassifierGui

MODELS_DIR = './models'

def available_model(path=MODELS_DIR):
    """Check if there is a model trained and ready. First application boot model will be trained using data"""
    return any(os.path.isfile(os.path.join(path, m) for m in os.listdir(path)))


if __name__ == "__main__":

    # if no model download and train the model

    app = FruitClassifierGui()