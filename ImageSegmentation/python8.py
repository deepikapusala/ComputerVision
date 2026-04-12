import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select your image file (e.g., the baboon or the mandrill image)...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load image in grayscale
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (300, 300))

        # --- SEGMENTATION TECHNIQUES ---

        # 1. Manual Single Thresholding (Histogram Based)
        # Every pixel above 127 becomes white (255), below becomes black (0)
        _, thresh_manual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 2. Automatic Thresholding (Otsu's Method)
        # As seen in your VLab screenshot - it calculates the best cutoff automatically
        _, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Edge-Based Segmentation (Canny Edge Detection)
        # Finds the boundaries of objects in the image
        edges = cv2.Canny(img, 100, 200)

        # --- DISPLAY RESULTS (1 Input + 3 Outputs) ---
        titles = ['Original Input', '1. Manual Threshold', 
                  "2. Otsu's Automatic", '3. Edge Segmentation']
        images = [img, thresh_manual, thresh_otsu, edges]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()