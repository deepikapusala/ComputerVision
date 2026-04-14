import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select your image file (e.g., fingerprint or text)...")
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

        # --- PRE-PROCESSING ---
        # Morphological ops work best on binary images. 
        # This converts your image to strict Black & White.
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Define the Structuring Element (Kernel)
        # Using a 5x5 square kernel as seen in your VLab screenshot
        kernel = np.ones((5, 5), np.uint8)

        # --- MORPHOLOGICAL OPERATIONS ---

        # 1. Erosion (Thins the image)
        erosion = cv2.erode(binary_img, kernel, iterations=1)

        # 2. Dilation (Thickens the image)
        dilation = cv2.dilate(binary_img, kernel, iterations=1)

        # 3. Opening (Erosion then Dilation - removes noise)
        opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

        # 4. Closing (Dilation then Erosion - fills gaps)
        closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

        # --- DISPLAY RESULTS (1 Input + 4 Outputs) ---
        titles = ['Original Binary', '1. Erosion', '2. Dilation', '3. Opening', '4. Closing']
        images = [binary_img, erosion, dilation, opening, closing]

        plt.figure(figsize=(18, 10))
        for i in range(5):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()