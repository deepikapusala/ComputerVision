import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image for PCA Compression...")
image_path = filedialog.askopenfilename(title="Select Image")

if image_path:
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
    else:
        img = cv2.resize(img, (400, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape to 2D for PCA
        h, w, c = img_rgb.shape
        img_reshaped = img_rgb.reshape(h, w * c)

        # --- PCA PROCESS ---
        # Keep only the first 30 components
        n_comp = 30
        pca = PCA(n_components=n_comp)
        compressed = pca.fit_transform(img_reshaped)
        reconstructed = pca.inverse_transform(compressed)
        
        # Reshape back to image format
        final_output = np.clip(reconstructed.reshape(h, w, c), 0, 255).astype(np.uint8)

        # --- DISPLAY ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("1. Original Input Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(final_output)
        plt.title(f"2. PCA Compressed ({n_comp} Components)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()