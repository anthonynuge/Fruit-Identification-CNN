import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk, ImageGrab
import uuid
import requests
from io import BytesIO
import os
import shutil

import sv_ttk

class FruitClassifierGui:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Fruit Classification App")
        self.root.geometry("1000x600")
        sv_ttk.set_theme("dark")
        self.canvas_width = 325
        self.canvas_height = 325

        # App data
        self.displayed_image = None
        self.images = []
        self.temp_dir = "./data/temp_images"

        # Bind Ctrl+V (or Command+V on macOS) to paste and save image
        self.root.bind("<Control-v>", self.paste_image)
        self.root.bind("<Command-v>", self.paste_image)  # For macOS compatibility
        os.makedirs(self.temp_dir, exist_ok=True)

        # Drag and drop 
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="#1C1C1C")
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

    def on_drop(self, event):
        """Handle drag-and-drop files"""
        # Check if files are dropped
        paths = self.root.tk.splitlist(event.data)
        for path in paths:
            if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.images.append(path)  # Add valid image file path
                self.display_image(-1)  # Display the latest image

    def paste_image(self):
        """Check the clipboard for an image and display it."""
        try:
            # Get the image from the clipboard
            clipboard_content = ImageGrab.grabclipboard()
            if isinstance(clipboard_content, Image.Image):
                # Display the image on the canvas
                self.display_image(clipboard_content)
            else:
                print("No image data found on clipboard.")
        except Exception as e:
            print(f"Error: {e}")

    def load_image(self):
        """
        Load a image or a dir of images into the app. First image is displayed
        """
        # Open a file dialog to select an image file
        if self.classification_mode.get() == 1 :
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            ) 

            if file_path:
                self.images.append(file_path)
                self.display_image(0)

        elif self.classification_mode.get() == 2 : 
            dir_path = filedialog.askdirectory()
            if dir_path:
                self.images = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                ]
                if self.images:
                    self.display_image(0)


    def display_image(self, index):
        if index < len(self.images):
            image = Image.open(self.images[index])
            # calculate scale factor to fit image within
            scale_factor = min(self.canvas_width / image.width, self.canvas_height / image.height)
            new_size = (int(image.width * scale_factor), (int(image.height * scale_factor)))
            resized_image = image.resize(new_size)
            self.displayed_image = ImageTk.PhotoImage(resized_image)
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, anchor="center", image=self.displayed_image)

    def paste_image(self, event=None):
        """Handle image pasting from the clipboard, checking for URLs or image data."""
        try:
            # Try to get clipboard content as text (URL case)
            clipboard_content = self.root.clipboard_get()
            if clipboard_content.startswith("http"):
                # If clipboard has a URL, try to download the image
                self.download_and_display_image(clipboard_content)
            else:
                # If clipboard doesn't contain a URL, try getting it as image data
                self.paste_image_from_clipboard()
        except tk.TclError:
            # Handle case where clipboard_get() fails and grab image directly
            self.paste_image_from_clipboard()

    def paste_image_from_clipboard(self):
        """Handle cases where an actual image is in the clipboard."""
        try:
            clipboard_content = ImageGrab.grabclipboard()
            if isinstance(clipboard_content, Image.Image):
                # Save image temporarily and display
                temp_path = self.save_image_temp(clipboard_content)
                self.images = [temp_path]
                self.display_image(0)
            else:
                print("No image data found on clipboard.")
        except Exception as e:
            print(f"Error grabbing image from clipboard: {e}")

    def download_and_display_image(self, url):
        """Download image from a URL and display it on the canvas."""
        try:
            response = requests.get(url)
            if response.status_code == 200 and "image" in response.headers["Content-Type"]:
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                temp_path = self.save_image_temp(image)
                self.image_paths.append(temp_path)
                self.display_image(temp_path)
            else:
                print("The URL does not contain an image.")
        except Exception as e:
            print(f"Error downloading image: {e}")

    def save_image_temp(self, image):
        """Save the image temporarily and return its file path."""
        image_name = f"{uuid.uuid4()}.png"
        temp_path = os.path.join(self.temp_dir, image_name)
        image.save(temp_path, "PNG")
        return temp_path
    
    def on_close(self):
        """Clean up temporary files and close the application."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)  # Delete the entire temp directory
                print(f"Temporary directory '{self.temp_dir}' has been cleared.")
        except Exception as e:
            print(f"Error clearing temporary files: {e}")
        finally:
            self.root.destroy()  # Close the application

    def test_stuff(self):
        # check radio button
        # res = "Single Mode" if self.classification_mode.get() == 1 else "BatchMode"
        # messagebox.showinfo("Selection", f"Current Mode: {res}")
        # get loaded images
        print(self.images)
