import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select your image for Geometric Transformations...")
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
        rows, cols, ch = img_rgb.shape

        # --- GEOMETRIC TRANSFORMATIONS ---

        # 1. Horizontal Shear
        # Matrix: [[1, sx, 0], [0, 1, 0]]
        M_shear_h = np.float32([[1, 0.5, 0], [0, 1, 0]])
        shear_h = cv2.warpAffine(img_rgb, M_shear_h, (int(cols*1.5), rows))

        # 2. Vertical Shear
        # Matrix: [[1, 0, 0], [sy, 1, 0]]
        M_shear_v = np.float32([[1, 0, 0], [0.5, 1, 0]])
        shear_v = cv2.warpAffine(img_rgb, M_shear_v, (cols, int(rows*1.5)))

        # 3. Reflection (Horizontal Flip)
        # 1 = horizontal, 0 = vertical, -1 = both
        reflection = cv2.flip(img_rgb, 1)

        # --- DISPLAY RESULTS ---
        titles = ['Original', 'Horizontal Shear', 'Vertical Shear', 'Reflection (Flip)']
        images = [img_rgb, shear_h, shear_v, reflection]

        plt.figure(figsize=(12, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()