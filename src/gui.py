import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import sv_ttk

class FruitClassifierGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Classification App")
        self.root.geometry("1000x600")

        # theme
        sv_ttk.set_theme("dark")

        # Canvas dimensions
        self.canvas_width = 325
        self.canvas_height = 325

        self.displayed_image = None
        self.current_image_path = None

        # classification mode tracker ( 1 = single file, 2 = batch )
        self.classification_mode = tk.IntVar(value=1)

        # Side bar frame ==========================================
        sidebar_frame = ttk.Frame(self.root, width=300)
        sidebar_frame.pack_propagate(False)
        sidebar_frame.pack(side="left", fill="y")

        sidebar_label =ttk.Label(sidebar_frame, text="Fruit Classifier", font=("Arial", 16))
        sidebar_label.pack(pady=10)

        mode_frame = ttk.Frame(sidebar_frame)
        mode_frame.pack(pady=10)

        # radio buttons for classification mode
        single_radio = ttk.Radiobutton(mode_frame, text="Single Image", variable=self.classification_mode, value=1)
        single_radio.pack(side="left", padx=5)
        batch_radio = ttk.Radiobutton(mode_frame, text="Multiple Images", variable=self.classification_mode, value=2)
        batch_radio.pack(side="right", padx=5)

        # load btn
        load_btn = ttk.Button(sidebar_frame, text="Load Image", command=self.load_image)
        load_btn.pack(pady=5, padx=10, fill="x")

        # Classify button
        classify_btn = ttk.Button(sidebar_frame, text="Classify Image")
        classify_btn.pack(pady=5, padx=10, fill="x")

        test_btn = ttk.Button(sidebar_frame, text="Test", command=self.test_stuff)
        test_btn.pack(pady=5, padx=10, fill="x")

        separator = ttk.Separator(self.root, orient="vertical")
        separator.pack(side="left", fill="y", padx=(0, 10))

        # main content frame ======================================
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side="right", expand=True, fill="both")

        # Canvas for displaying the selected image
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="gray30")
        self.canvas.pack(pady=20)
        self.canvas.create_text(self.canvas_width // 2, self.canvas_height // 2, text="No Image Loaded", fill="white")

        horizonal_separator = ttk.Separator(main_frame, orient="horizontal")
        horizonal_separator.pack(fill="x", pady=5)

        # result frame ============================================
        result_frame = ttk.Frame(main_frame, padding=10)
        result_frame.pack(fill="x", pady=(5, 2))

        result_label = ttk.Label(result_frame, text="Results", font=("Arial", 14))
        result_label.pack(anchor="w")

        result_text = ttk.Label(result_frame, text="N/a")
        result_text.pack(anchor="w")


        self.root.mainloop()


    def load_image(self):
        """
        Load an image from file and display it in the canvas.
        """
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            # Load and resize the image to fit within the canvas
            image = Image.open(file_path)

            # calculate scale factor to fit image within
            scale_factor = min(self.canvas_width / image.width, self.canvas_height / image.height)

            new_size = (int(image.width * scale_factor), (int(image.height * scale_factor)))
            resized_image = image.resize(new_size)
            self.displayed_image = ImageTk.PhotoImage(resized_image)

            self.canvas.delete("all")
            self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, anchor="center", image=self.displayed_image)

    def test_stuff(self):
        res = "Single Mode" if self.classification_mode.get() == 1 else "BatchMode"
        messagebox.showinfo("Selection", f"Current Mode: {res}")


