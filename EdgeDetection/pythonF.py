import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select an image to test Canny Edge Detection...")
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
        img = cv2.resize(img, (400, 400))

        # --- CANNY EDGE DETECTION STAGES ---

        # Step 1: Noise Reduction (Gaussian Blur is built-in or done manually)
        blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

        # Step 2: Canny with different thresholds
        # Low Threshold: 50, High Threshold: 150 (Standard)
        canny_std = cv2.Canny(blurred, 50, 150)

        # Wide Threshold (Captures more noise/details)
        canny_wide = cv2.Canny(blurred, 10, 200)

        # Tight Threshold (Captures only the strongest edges)
        canny_tight = cv2.Canny(blurred, 150, 250)

        # --- DISPLAY RESULTS ---
        titles = ['Original Grayscale', 'Standard Canny (50, 150)', 
                  'Wide (10, 200)', 'Tight (150, 250)']
        images = [img, canny_std, canny_wide, canny_tight]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()