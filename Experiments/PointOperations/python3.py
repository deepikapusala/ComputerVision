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
    # Load image in grayscale as point operations focus on intensity values
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (300, 300))

        # --- POINT OPERATIONS ---

        # 1. Image Negation (Reversing Polarity: Dark to White)
        # Formula: s = (L - 1) - r
        img_negation = 255 - img

        # 2. Log Transformation (Expands dark pixels)
        # Formula: s = c * log(1 + r)
        c = 255 / np.log(1 + np.max(img))
        log_transformed = c * (np.log(img + 1))
        log_transformed = np.array(log_transformed, dtype=np.uint8)

        # 3. Linear Mapping / Contrast Stretching (f(r) = mr + c)
        # Similar to your VLab screenshot: f(r) = -0.38r + 59
        # We'll do a simple contrast stretch here
        xp = [np.min(img), np.max(img)]
        fp = [0, 255]
        img_linear = np.interp(img, xp, fp).astype('uint8')

        # --- DISPLAY RESULTS ---
        titles = ['Original Input', '1. Negation (Polarity)', 
                  '2. Log Transform', '3. Linear Mapping']
        images = [img, img_negation, log_transformed, img_linear]

        plt.figure(figsize=(12, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()