import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image to see the Boundary Extraction...")
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
        
        # --- PRE-PROCESSING ---
        # Convert to grayscale then Binary
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Define Structuring Element (Kernel)
        # 3x3 or 5x5 works well
        kernel = np.ones((3, 3), np.uint8)

        # --- BOUNDARY EXTRACTION ALGORITHM ---
        # Step 1: Erode the image (shrinks it slightly)
        eroded = cv2.erode(binary_img, kernel, iterations=1)

        # Step 2: Subtract the eroded image from the original
        # Formula: Boundary(A) = A - (A eroded by B)
        boundary = cv2.subtract(binary_img, eroded)

        # --- DISPLAY RESULTS (4 Images) ---
        titles = ['1. Original Input', '2. Binary Version', 
                  '3. Eroded Image', '4. Extracted Boundary']
        images = [img_rgb, binary_img, eroded, boundary]

        plt.figure(figsize=(12, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            if i == 0:
                plt.imshow(images[i])
            else:
                plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()