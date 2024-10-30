import tkinter as tk
from PIL import Image, ImageTk

import sv_ttk


def start_gui():
    """
    Start the gui
    """
    root = tk.Tk()
    root.title("Fruit Classifier App")
    root.geometry("1200x600")

    sv_ttk.use_dark_theme()

    # Side bar frame
    sidebar_frame = tk.Frame(root, width=200)
    sidebar_frame.pack(side="left", fill="y")

    # main content frame
    main_frame = tk.Frame(root)
    main_frame.pack(side="right", expand=True, fill="both")

    # Result frame
    result_frame = tk.Frame(root)
    result_frame.pack(side="bottom", fill="x", padx=10, pady=10)

    # Buttons for sidebar
    load_btn = tk.Button(sidebar_frame, text="Load Image")
    load_btn.pack(pady=10, padx=10, fill="x")

    label = tk.Label(root, text="Fruit Classifier is Ready!", font=("Arial", 16))
    label.pack(pady=20)

    button = tk.Button(root, text="Click me!")
    button.pack(pady=20)

    root.mainloop()
