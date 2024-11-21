import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk, ImageGrab
import uuid
import requests
from io import BytesIO
import os
import shutil
from src.model import predict_image_confidence
from src.loader import load_and_preprocess_image
from collections import deque

import sv_ttk

class ImageData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.top_pred = "N/A"
        self.top_conf = 00.00
        self.confidence_report = []

    def set_report(self, sorted_report):
        self.confidence_report = sorted_report
        self.top_pred, self.top_conf = sorted_report[0]

class FruitClassifierGui:
    def __init__(self, model=None, class_names=[]):
        self.root = TkinterDnD.Tk()
        self.root.title("Fruit Classification App")
        self.root.geometry("1000x600")
        sv_ttk.set_theme("dark")
        self.canvas_width = 325
        self.canvas_height = 325

        # state variables
        self.displayed_image = None
        self.images = []
        self.temp_dir = "./data/temp_images"
        self.model = model
        self.class_names = class_names
        # classification mode tracker ( 1 = single file, 2 = batch )
        self.classification_mode = tk.IntVar(value=1) 
        self.images_dict = {}
        self.current_image = None

        # Bind Ctrl+V (or Command+V on macOS) to paste and save image
        # self.root.bind("<Control-v>", self.paste_image)
        # self.root.bind("<Command-v>", self.paste_image)  
        # os.makedirs(self.temp_dir, exist_ok=True)

        # Drag and drop 
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Side bar frame ==========================================
        sidebar_frame = ttk.Frame(self.root, width=300)
        sidebar_frame.pack_propagate(False)
        sidebar_frame.pack(side="left", fill="y")

        sidebar_label =ttk.Label(sidebar_frame, text="Fruit Classifier", font=("Roboto", 16, "bold"))
        sidebar_label.pack(pady=10)

        mode_frame = ttk.Frame(sidebar_frame)
        mode_frame.pack(pady=10)

        # radio buttons for classification mode
        single_radio = ttk.Radiobutton(mode_frame, text="Single", variable=self.classification_mode, value=1)
        single_radio.pack(side="left", padx=5)
        batch_radio = ttk.Radiobutton(mode_frame, text="Batch", variable=self.classification_mode, value=2)
        batch_radio.pack(side="right", padx=5)

        # load btn
        load_btn = ttk.Button(sidebar_frame, text="Load Image", command=self.load_image)
        load_btn.pack(pady=5, padx=10, fill="x")

        # Classify button
        classify_btn = ttk.Button(sidebar_frame, text="Classify", command=self.handle_classify)
        classify_btn.pack(pady=5, padx=10, fill="x")

        # Clear button
        clear_btn = ttk.Button(sidebar_frame, text="Clear", command=self.clear)
        clear_btn.pack(pady=5, padx=10, fill="x")

        self.loaded_images_table = ttk.Treeview(sidebar_frame, columns=("File", "Prediction"), show="headings")
        self.loaded_images_table.heading("File", text="File")
        self.loaded_images_table.heading("Prediction", text="Prediction")
        self.loaded_images_table.column("File", anchor="w", width=100)
        self.loaded_images_table.column("Prediction", anchor="center", width=100)
        self.loaded_images_table.pack(pady=5, padx=10, fill="both", expand=True)

        self.loaded_images_table.bind("<<TreeviewSelect>>", self.table_select)

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
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        result_frame_label = ttk.Label(result_frame, text="OUTPUT: ", font=("Roboto", 12, "bold"))
        result_frame_label.pack(anchor="w")

        analysis_frame = ttk.Frame(result_frame)
        analysis_frame.pack(fill="both", expand=True)
        analysis_frame.rowconfigure(0, weight=1)
        analysis_frame.columnconfigure(0, weight=2)
        analysis_frame.columnconfigure(1, weight=3)

        prediction_frame = ttk.Frame(analysis_frame)
        prediction_frame.grid(row=0, column=0, sticky="nswe")

        self.classification_report_frame = ttk.Frame(analysis_frame)
        self.classification_report_frame.grid(row=0, column=1, sticky="nswe")

        self.prediction_percentage_label = ttk.Label(
            prediction_frame, text = "00.00%", font=("Roboto", 28, "bold")
        )
        self.prediction_percentage_label.pack(pady=(50, 5), anchor="center")
        self.prediction_label = ttk.Label(
            prediction_frame, text = "N/A", font=("Roboto", 16)
        )
        self.prediction_label.pack(pady=(5,20), anchor="center")

        table_frame = ttk.Frame(self.classification_report_frame)
        table_frame.pack(fill="both", expand=True)

        self.report_table = ttk.Treeview(table_frame, columns=("Class", "Confidence"), show="headings")
        self.report_table.heading("Class", text="Class")
        self.report_table.heading("Confidence", text="Confidence (%)")
        self.report_table.column("Class", width=200, anchor="center")
        self.report_table.column("Confidence", width=75, anchor="center")
        self.report_table.pack(side="left", fill="both", expand=True)
        # Add scrollbar for the table
        scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.report_table.yview
        )
        self.report_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        self.root.mainloop()

    def handle_classify(self):
        mode = self.classification_mode.get()
        if mode == 1:
            self.classify_image()
        elif mode == 2:
            self.classify_all()

    def classify_image(self):
        if self.model is None:
            print("No model loaded.")
            return
        if self.current_image is None:
            print("No image loaded.")
            return 

        image_data = self.images_dict[self.current_image]

        if (image_data.top_pred != "N/A"):
            print("Already classified")
            return 

        processed_image = load_and_preprocess_image(image_data.file_path)
        result, confidence_report = predict_image_confidence(self.model, processed_image, self.class_names)

        classification_report_percentages = {
            class_name: round(confidence * 100, 2) for class_name, confidence in confidence_report.items()
        }
        sorted_report = sorted(classification_report_percentages.items(), key=lambda x: x[1], reverse=True)
        image_data.set_report(sorted_report)
        self.update_results(image_data)
        self.update_loaded_images_table()
    
    def classify_all(self):
        if not self.images_dict:
            print("No images loaded")
            return
        
        for file_path, image_data in self.images_dict.items():

            processed_image = load_and_preprocess_image(file_path)
            result, confidence_report = predict_image_confidence(self.model, processed_image, self.class_names)

            classification_report_percentages = {
                class_name: round(confidence * 100, 2) for class_name, confidence in confidence_report.items()
            }
            sorted_report = sorted(classification_report_percentages.items(), key=lambda x: x[1], reverse=True)
            image_data.set_report(sorted_report)
        self.update_loaded_images_table()


    def update_results(self, image_data):

        if image_data.top_conf == 00.00 and image_data.top_pred == "N/A":
            color = "white"
        elif image_data.top_conf > 90:
            color = "green"
        elif image_data.top_conf > 70:
            color = "yellow"
        else:
            color = "red"
        self.clear_table()
        for class_name, confidence in image_data.confidence_report[1:]:
            self.report_table.insert("", "end", values=(class_name, f"{confidence}%"))
        self.prediction_percentage_label.config(text=f"{image_data.top_conf:.2f}%", foreground=color)
        self.prediction_label.config(text=f"{image_data.top_pred}", foreground=color)


    def update_loaded_images_table(self):
        for row in self.loaded_images_table.get_children():
            self.loaded_images_table.delete(row)
        
        for file_path, image_item in self.images_dict.items():
            self.loaded_images_table.insert(
                "",
                "end",
                iid=file_path,
                values=(
                    image_item.filename,
                    image_item.top_pred,
                )
            )

    def clear(self): 
        self.current_image = ""
        self.displayed_image = None
        self.images_dict = {}
        self.update_loaded_images_table()
        self.clear_results()

    def clear_table(self): 
        for row in self.report_table.get_children():
            self.report_table.delete(row)
    
    def clear_results(self):
        self.prediction_label.config(text="N/A", foreground="white")
        self.prediction_percentage_label.config(text="00.00%", foreground="white")
        self.clear_table()

    def table_select(self, event):
        selected = self.loaded_images_table.selection()

        if not selected:
            return

        file_path = selected[0]
        print("Selected from tree: ", file_path)

        if file_path in self.images_dict:
            image_data = self.images_dict[file_path]
            self.display_image(file_path)
            self.clear_results()
            self.update_results(image_data)
            self.current_image = file_path


    def on_drop(self, event):
        """Handle drag-and-drop files"""
        # Check if files are dropped
        paths = self.root.tk.splitlist(event.data)
        for file_path in paths:
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                new_image = ImageData(file_path)
                self.images_dict[file_path] = new_image
                self.update_loaded_images_table()
                self.loaded_images_table.selection_set(file_path)
                self.loaded_images_table.see(file_path)
                self.display_image(file_path)
                self.current_image = file_path

    # def paste_image(self):
    #     """Check the clipboard for an image and display it."""
    #     try:
    #         # Get the image from the clipboard
    #         clipboard_content = ImageGrab.grabclipboard()
    #         if isinstance(clipboard_content, Image.Image):
    #             # Display the image on the canvas
    #             self.display_image(clipboard_content)
    #         else:
    #             print("No image data found on clipboard.")
    #     except Exception as e:
    #         print(f"Error: {e}")

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
                new_image = ImageData(file_path)
                self.images_dict[file_path] = new_image
                self.update_loaded_images_table()
                self.loaded_images_table.selection_set(file_path)
                self.loaded_images_table.see(file_path)
                self.display_image(file_path)
                self.current_image = file_path

        elif self.classification_mode.get() == 2 : 
            dir_path = filedialog.askdirectory()
            if dir_path:
                for f in os.listdir(dir_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        file_path = os.path.join(dir_path, f)
                        if file_path not in self.images_dict:
                            self.images_dict[file_path] = ImageData(file_path)
                self.update_loaded_images_table()

    def display_image(self, file_path):
            image = Image.open(file_path)
            # calculate scale factor to fit image within
            scale_factor = min(self.canvas_width / image.width, self.canvas_height / image.height)
            new_size = (int(image.width * scale_factor), (int(image.height * scale_factor)))
            resized_image = image.resize(new_size)
            self.displayed_image = ImageTk.PhotoImage(resized_image)
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, anchor="center", image=self.displayed_image)

    # def paste_image(self, event=None):
    #     """Handle image pasting from the clipboard, checking for URLs or image data."""
    #     try:
    #         # Try to get clipboard content as text (URL case)
    #         clipboard_content = self.root.clipboard_get()
    #         if clipboard_content.startswith("http"):
    #             # If clipboard has a URL, try to download the image
    #             self.download_and_display_image(clipboard_content)
    #         else:
    #             # If clipboard doesn't contain a URL, try getting it as image data
    #             self.paste_image_from_clipboard()
    #     except tk.TclError:
    #         # Handle case where clipboard_get() fails and grab image directly
    #         self.paste_image_from_clipboard()

    # def paste_image_from_clipboard(self):
    #     """Handle cases where an actual image is in the clipboard."""
    #     try:
    #         clipboard_content = ImageGrab.grabclipboard()
    #         if isinstance(clipboard_content, Image.Image):
    #             # Save image temporarily and display
    #             temp_path = self.save_image_temp(clipboard_content)
    #             self.load_image(temp_path)
    #             # self.images = [temp_path]
    #             # self.display_image(0)
    #         else:
    #             print("No image data found on clipboard.")
    #     except Exception as e:
    #         print(f"Error grabbing image from clipboard: {e}")

    # def download_and_display_image(self, url):
    #     """Download image from a URL and display it on the canvas."""
    #     try:
    #         response = requests.get(url)
    #         if response.status_code == 200 and "image" in response.headers["Content-Type"]:
    #             image_data = BytesIO(response.content)
    #             image = Image.open(image_data)
    #             temp_path = self.save_image_temp(image)
    #             self.image_paths.append(temp_path)
    #             self.display_image(temp_path)
    #         else:
    #             print("The URL does not contain an image.")
    #     except Exception as e:
    #         print(f"Error downloading image: {e}")

    # def save_image_temp(self, image):
    #     """Save the image temporarily and return its file path."""
    #     image_name = f"{uuid.uuid4()}.png"
    #     temp_path = os.path.join(self.temp_dir, image_name)
    #     image.save(temp_path, "PNG")
    #     return temp_path
    
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

