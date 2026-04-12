import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select an image to see the Pyramid (Wavelet) levels...")
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
        img = cv2.resize(img, (512, 512)) # Power of 2 sizes work best for pyramids
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- GAUSSIAN PYRAMID (Downsampling) ---
        # Level 0: Original
        layer = img_rgb.copy()
        gaussian_pyr = [layer]
        
        for i in range(2):
            layer = cv2.pyrDown(layer) # Shrinks image by half
            gaussian_pyr.append(layer)

        # --- LAPLACIAN PYRAMID (Detail Extraction) ---
        # This shows the "high-frequency" details (edges) at each level
        # Formula: L = G_n - Expand(G_n+1)
        
        # Level 1 Detail
        upper = cv2.pyrUp(gaussian_pyr[1]) # Upsample the smaller one
        # Resize upper to match layer 0 exactly (handles rounding issues)
        upper = cv2.resize(upper, (gaussian_pyr[0].shape[1], gaussian_pyr[0].shape[0]))
        laplacian_1 = cv2.subtract(gaussian_pyr[0], upper)

        # --- DISPLAY RESULTS ---
        plt.figure(figsize=(15, 10))

        # 1. Original (Level 0)
        plt.subplot(2, 2, 1)
        plt.imshow(gaussian_pyr[0])
        plt.title('Gaussian Level 0 (Original)')
        plt.axis('off')

        # 2. Level 1 (Half size)
        plt.subplot(2, 2, 2)
        plt.imshow(gaussian_pyr[1])
        plt.title('Gaussian Level 1 (1/2 Scale)')
        plt.axis('off')

        # 3. Level 2 (Quarter size)
        plt.subplot(2, 2, 3)
        plt.imshow(gaussian_pyr[2])
        plt.title('Gaussian Level 2 (1/4 Scale)')
        plt.axis('off')

        # 4. Laplacian Detail (The 'Wavelet-like' edge info)
        plt.subplot(2, 2, 4)
        plt.imshow(laplacian_1)
        plt.title('Laplacian Detail (Edges)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()