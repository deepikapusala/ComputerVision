import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.signal import wiener

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select an image to test the Wiener Filter...")
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
        img = cv2.resize(img, (400, 400))

        # --- STEP 1: CREATE DEGRADATION ---
        # 1. Add Motion-like Blur
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        blurred = cv2.filter2D(img, -1, kernel)
        
        # 2. Add Gaussian Noise
        noise = np.random.normal(0, 20, img.shape).astype('int16')
        degraded = np.clip(blurred.astype('int16') + noise, 0, 255).astype('uint8')

        # --- STEP 2: WIENER FILTER RESTORATION ---
        # The 'mysize' parameter is the window size for local variance estimation
        # We test two different window sizes to see the effect
        restored_w3 = wiener(degraded.astype('float32'), mysize=3)
        restored_w3 = np.clip(restored_w3, 0, 255).astype('uint8')
        
        restored_w7 = wiener(degraded.astype('float32'), mysize=7)
        restored_w7 = np.clip(restored_w7, 0, 255).astype('uint8')

        # --- DISPLAY RESULTS ---
        titles = ['1. Degraded (Blur + Noise)', '2. Wiener (3x3 Window)', 
                  '3. Wiener (7x7 Window)', '4. Original Reference']
        images = [degraded, restored_w3, restored_w7, img]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()