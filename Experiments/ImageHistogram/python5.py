import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    # Load image in grayscale
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Error: Could not load image.")
    else:
        # Standardize size for the 4-quadrant split
        img = cv2.resize(img, (300, 300))
        h, w = img.shape

        # --- SUB-IMAGE CALCULATIONS (Quadrants 1, 2, 3, 4) ---
        # Splitting the image into four equal parts
        # Quadrant 1: Top-Left  | Quadrant 2: Top-Right
        # Quadrant 3: Bottom-Left| Quadrant 4: Bottom-Right
        mid_h, mid_w = h // 2, w // 2
        
        sub1 = img[0:mid_h, 0:mid_w]
        sub2 = img[0:mid_h, mid_w:w]
        sub3 = img[mid_h:h, 0:mid_w]
        sub4 = img[mid_h:h, mid_w:w]

        # --- DISPLAY RESULTS ---
        plt.figure(figsize=(15, 10))

        # 1. Original Image
        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Source Image')
        plt.axis('off')

        # 2. Full Image Histogram
        plt.subplot(2, 3, 2)
        plt.hist(img.ravel(), 256, [0, 256], color='gold')
        plt.title('Full Image Histogram')

        # 3. Sub-histogram 1
        plt.subplot(2, 3, 3)
        plt.hist(sub1.ravel(), 256, [0, 256], color='yellow')
        plt.title('Sub-histogram 1 (Top-Left)')

        # 4. Sub-histogram 2
        plt.subplot(2, 3, 4)
        plt.hist(sub2.ravel(), 256, [0, 256], color='yellow')
        plt.title('Sub-histogram 2 (Top-Right)')

        # 5. Sub-histogram 3
        plt.subplot(2, 3, 5)
        plt.hist(sub3.ravel(), 256, [0, 256], color='yellow')
        plt.title('Sub-histogram 3 (Bottom-Left)')

        # 6. Sub-histogram 4
        plt.subplot(2, 3, 6)
        plt.hist(sub4.ravel(), 256, [0, 256], color='yellow')
        plt.title('Sub-histogram 4 (Bottom-Right)')

        plt.tight_layout()
        plt.show()