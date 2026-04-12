import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# 1. Open a local file picker window
root = tk.Tk()
root.withdraw()
print("Select your image file in the popup window...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image.")
    else:
        # Standardize size
        img = cv2.resize(img, (300, 300))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- NEIGHBOURHOOD OPERATIONS (FILTERS) ---

        # 1. Average Filter (Box Blur) - Linear
        # Uses a 5x5 kernel where all pixels have equal weight
        avg_blur = cv2.blur(img_rgb, (5, 5))

        # 2. Gaussian Filter - Linear
        # Pixels closer to the center have more weight (smoother blur)
        gaussian_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)

        # 3. Median Filter - Non-Linear
        # Excellent for salt-and-pepper noise removal
        # Replaces pixel with the median value of its neighborhood
        median_blur = cv2.medianBlur(img_rgb, 5)

        # --- DISPLAY RESULTS (1 Input + 3 Outputs) ---
        titles = ['Original Input', '1. Average Filter', 
                  '2. Gaussian Filter', '3. Median Filter']
        images = [img_rgb, avg_blur, gaussian_blur, median_blur]

        plt.figure(figsize=(12, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()