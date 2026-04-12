import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog # THIS was the missing piece!

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image for Bayer Demosaicing...")
image_path = filedialog.askopenfilename(title="Select Colored Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected!")
else:
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
    else:
        img = cv2.resize(img, (400, 400))
        
        # 1. Simulate a Bayer BG Pattern (Raw Sensor Data)
        # Digital sensors capture a grid of R, G, and B
        raw_bayer = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Simplified Bayer BG grid simulation
        raw_bayer[0::2, 0::2] = img[0::2, 0::2, 0] # Blue
        raw_bayer[0::2, 1::2] = img[0::2, 1::2, 1] # Green
        raw_bayer[1::2, 0::2] = img[1::2, 0::2, 1] # Green
        raw_bayer[1::2, 1::2] = img[1::2, 1::2, 2] # Red

        # 2. Demosaic (Reconstruct Color from the raw grid)
        # cv2.COLOR_BayerBG2RGB fills in the missing colors for each pixel
        reconstructed = cv2.cvtColor(raw_bayer, cv2.COLOR_BayerBG2RGB)

        # --- DISPLAY RESULTS (3-Step View) ---
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("1. Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(raw_bayer, cmap='gray')
        plt.title("2. Raw Bayer Pattern (Sensor View)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed)
        plt.title("3. Demosaiced Reconstruction")
        plt.axis('off')

        plt.tight_layout()
        plt.show()