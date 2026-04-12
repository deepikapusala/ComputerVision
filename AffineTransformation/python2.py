import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# 1. Open a local file picker window
root = tk.Tk()
root.withdraw() # Hide the tiny empty tkinter window
print("Select your image file in the popup window...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image. Check the file format.")
    else:
        # Standardize size
        img = cv2.resize(img, (400, 400))
        rows, cols, ch = img.shape

        # --- AFFINE TRANSFORMATIONS ---
        
        # 1. Translation (Shifting)
        M_translate = np.float32([[1, 0, 70], [0, 1, 40]])
        translated = cv2.warpAffine(img, M_translate, (cols, rows))

        # 2. Rotation (90 Degrees)
        center = (cols // 2, rows // 2)
        M_rotate = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

        # 3. Scaling (Zooming 1.5x)
        scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # --- DISPLAY RESULTS ---
        titles = ['Original Input', '1. Translated (Shifted)', 
                  '2. Rotated (90 deg)', '3. Scaled (Zoomed)']
        
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(translated, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)]

        plt.figure(figsize=(12, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()