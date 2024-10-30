import os
import subprocess
from tkinter import messagebox

dataset = "moltean/fruits"
download_dir = "data/raw"
kaggle_credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")


def check_credentials():
    """
    Check if kaggle api credential present
    """
    if not os.path.exists(kaggle_credentials_path):
        messagebox.showerror(
            "Missing Credientials for Kaggle",
            f"Kaggle API credentials not found. Please place your kaggle.json file in {kaggle_credentials_path}",
        )
        return False
    return True


def download_dataset():
    """
    Download dataset if it is not present using the kaggle api
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    if len(os.listdir(download_dir)) > 0:
        print(f"Dataset already exists in {download_dir}")
        return

    if not check_credentials():
        return

    try:
        print(f"Download dataset: {dataset}")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "--unzip",
                "-p",
                download_dir,
            ],
            check=True,
        )
        print(f"Dataset download and extracted to {download_dir}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "Download Error",
            "Failed to download the dataset. Please check your internet connection and Kaggle credentials.",
        )
        print(f"Error downloading: {e}")
