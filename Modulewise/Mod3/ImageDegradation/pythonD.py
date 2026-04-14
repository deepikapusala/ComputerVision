import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.signal import wiener

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select an image to degrade and restore...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load image in grayscale for easier restoration analysis
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (400, 400))

        # --- STEP 1: DEGRADATION (Add Noise) ---
        # Modeling a noisy environment
        noise = np.random.normal(0, 30, img.shape).astype('int16')
        degraded = np.clip(img.astype('int16') + noise, 0, 255).astype('uint8')

        # --- STEP 2: RESTORATION (Mean Filter) ---
        # Simple restoration using local averaging
        restored_mean = cv2.blur(degraded, (5, 5))

        # --- STEP 3: RESTORATION (Wiener Filter) ---
        # Advanced restoration that minimizes mean square error
        # (Requires scipy.signal.wiener)
        restored_wiener = wiener(degraded.astype('float32'), mysize=5)
        restored_wiener = np.clip(restored_wiener, 0, 255).astype('uint8')

        # --- DISPLAY RESULTS ---
        titles = ['Original (Clear)', 'Degraded (Noise)', 
                  'Restored (Mean Filter)', 'Restored (Wiener)']
        images = [img, degraded, restored_mean, restored_wiener]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()