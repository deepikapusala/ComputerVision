import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image for the fixed HDR Imaging...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected!")
else:
    # Load image as 8-bit BGR
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
    else:
        img = cv2.resize(img, (400, 400))

        # --- SIMULATING HDR EXPOSURES (Staying in 8-bit) ---
        # We use cv2.convertScaleAbs to keep the depth at CV_8U
        dark = cv2.convertScaleAbs(img, alpha=0.5, beta=0)   # Underexposed
        bright = cv2.convertScaleAbs(img, alpha=1.5, beta=0) # Overexposed
        
        # List of images and exposure times
        img_list = [dark, img, bright]
        exposure_times = np.array([0.5, 1.0, 1.5], dtype=np.float32)

        # --- MERGE TO HDR ---
        # This part will now pass the 'Assertion failed' check
        merge_debevec = cv2.createMergeDebevec()
        hdr = merge_debevec.process(img_list, times=exposure_times)

        # Tonemap back to 8-bit so it's viewable
        tonemap = cv2.createTonemap(gamma=2.2)
        res_debevec = tonemap.process(hdr)
        res_8bit = np.clip(res_debevec * 255, 0, 255).astype(np.uint8)

        # --- DISPLAY ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("1. Original Input")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(res_8bit, cv2.COLOR_BGR2RGB))
        plt.title("2. Fixed HDR Result")
        plt.axis('off')

        plt.tight_layout()
        plt.show()