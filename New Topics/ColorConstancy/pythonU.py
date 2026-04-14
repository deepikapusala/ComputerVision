import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog # FIXED: Explicitly imported for the file picker

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image for Color Constancy...")
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
        
        # --- GRAY WORLD ALGORITHM ---
        # 1. Calculate average for each channel
        # We use float32 to avoid overflow during math
        img_float = img_rgb.astype(np.float32)
        avg_r = np.mean(img_float[:, :, 0])
        avg_g = np.mean(img_float[:, :, 1])
        avg_b = np.mean(img_float[:, :, 2])
        
        # 2. Calculate the overall gray mean
        avg_gray = (avg_r + avg_g + avg_b) / 3
        
        # 3. Scale each channel so its average equals the overall mean
        img_corrected = img_float.copy()
        img_corrected[:, :, 0] *= (avg_gray / avg_r)
        img_corrected[:, :, 1] *= (avg_gray / avg_g)
        img_corrected[:, :, 2] *= (avg_gray / avg_b)
        
        # 4. Clip values to [0, 255] and convert back to uint8
        img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)

        # --- DISPLAY RESULTS (Input included) ---
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("1. Original Input (With Tint)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_corrected)
        plt.title("2. Color Constancy (Gray World Fix)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()