import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image to see the Color Image Smoothing and Sharpening...")
image_path = filedialog.askopenfilename(title="Select Colored Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load colored image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (400, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- 1. COLOR SMOOTHING (Gaussian Blur) ---
        # Reduces noise and detail by averaging pixel values
        smoothed = cv2.GaussianBlur(img_rgb, (11, 11), 0)

        # --- 2. COLOR SHARPENING (Unsharp Masking) ---
        # We use a Laplacian kernel to find edges and add them back to the image
        kernel_sharpening = np.array([[-1,-1,-1], 
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
        sharpened = cv2.filter2D(img_rgb, -1, kernel_sharpening)

        # --- 3. ADVANCED SHARPENING (L-channel only) ---
        # Best practice: process only Lightness in LAB space to avoid color noise
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l_sharpened = cv2.filter2D(l, -1, kernel_sharpening)
        refined_lab = cv2.merge((l_sharpened, a, b))
        refined_sharpened = cv2.cvtColor(refined_lab, cv2.COLOR_Lab2RGB)

        # --- DISPLAY RESULTS ---
        titles = ['Original Image', '1. Smoothed (Blur)', 
                  '2. Basic Sharpen (RGB)', '3. Refined Sharpen (LAB)']
        images = [img_rgb, smoothed, sharpened, refined_sharpened]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()