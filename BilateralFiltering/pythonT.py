import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog # FIXED: Explicit import

# 1. Open local file picker
root = tk.Tk(); root.withdraw()
print("Select a COLORED image for Bilateral Filtering...")
path = filedialog.askopenfilename(title="Select Image", 
                                  filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not path:
    print("No file selected!")
else:
    img = cv2.imread(path)
    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (400, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Standard Gaussian Blur (Blurs edges)
        gaussian = cv2.GaussianBlur(img_rgb, (15, 15), 0)

        # Bilateral Filter (Smooths flat areas, keeps edges SHARP)
        bilateral = cv2.bilateralFilter(img_rgb, 15, 75, 75)

        # --- DISPLAY (3 Images) ---
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("1. Original Input")
        plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(gaussian); plt.title("2. Gaussian (Edges Blurry)")
        plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(bilateral); plt.title("3. Bilateral (Edges Sharp)")
        plt.axis('off')
        
        plt.tight_layout(); plt.show()