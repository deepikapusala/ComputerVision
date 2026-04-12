import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select your image for the Test Bench (e.g., the car or the mandrill)...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
    else:
        # Standardize for the bench
        img = cv2.resize(img, (400, 400))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- THE PIPELINE (Test Bench) ---

        # PIPIELINE STEP 1: Enhancement
        # Histogram Equalization to bring out details
        enhanced = cv2.equalizeHist(gray)

        # PIPELINE STEP 2: Neighbourhood Operation
        # Median Blur to remove any salt-and-pepper noise
        denoised = cv2.medianBlur(enhanced, 5)

        # PIPELINE STEP 3: Segmentation (The Final Goal)
        # Using Otsu's Automatic Thresholding as seen in your screenshot
        _, segmented = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- DISPLAY RESULTS (The 'Piping' View) ---
        titles = ['1. Source Image', '2. Enhanced (Histogram)', 
                  '3. Denoised (Median)', '4. Final Segmentation']
        images = [gray, enhanced, denoised, segmented]

        plt.figure(figsize=(16, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()