from src.download_dataset import download_dataset
import tkinter as tk
from src.gui import FruitClassifierGui

if __name__ == "__main__":

    # if no model download and train the model

    root = tk.Tk()
    root.style = tk.ttk.Style()
    root.style.theme_use("clam")


    app = FruitClassifierGui(root)


