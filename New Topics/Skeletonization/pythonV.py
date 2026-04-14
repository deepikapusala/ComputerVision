import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk(); root.withdraw()
print("Select a B&W or Binary image for Skeletonization...")
path = filedialog.askopenfilename(title="Select Image")

if path:
    img = cv2.imread(path, 0) # Load as grayscale
    img = cv2.resize(img, (300, 300))
    # Ensure it's binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # --- SKELETONIZATION ALGORITHM ---
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp_img = binary.copy()
    
    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break

    # --- DISPLAY ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(binary, cmap='gray'); plt.title("1. Original Binary")
    plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(skel, cmap='gray'); plt.title("2. Skeletonization")
    plt.axis('off')
    plt.tight_layout(); plt.show()