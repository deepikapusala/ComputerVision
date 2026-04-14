import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select your image to test the Gaussian Filter...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (400, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- GAUSSIAN FILTERING (Linear) ---
        
        # Kernel size must be ODD (3x3, 5x5, 7x7, etc.)
        # The third parameter (0) tells OpenCV to calculate the 'sigma' (spread) automatically
        
        # 1. Mild Blur (3x3 Kernel)
        gaussian_mild = cv2.GaussianBlur(img_rgb, (3, 3), 0)

        # 2. Moderate Blur (7x7 Kernel)
        gaussian_med = cv2.GaussianBlur(img_rgb, (7, 7), 0)

        # 3. Heavy Blur (15x15 Kernel)
        gaussian_heavy = cv2.GaussianBlur(img_rgb, (15, 15), 0)

        # --- DISPLAY RESULTS ---
        titles = ['Original Image', 'Gaussian Blur (3x3)', 
                  'Gaussian Blur (7x7)', 'Gaussian Blur (15x15)']
        images = [img_rgb, gaussian_mild, gaussian_med, gaussian_heavy]

        plt.figure(figsize=(12, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()