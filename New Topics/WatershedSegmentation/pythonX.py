import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

root = tk.Tk(); root.withdraw()
print("Select a COLORED image (e.g., coins/fruit) for Watershed...")
path = filedialog.askopenfilename()

if path:
    img = cv2.imread(path)
    img = cv2.resize(img, (400, 400))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to find foreground/background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Sure foreground area (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0] # Mark boundaries in Red

    # --- DISPLAY ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("1. Original Input")
    plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(thresh, cmap='gray'); plt.title("2. Binary Threshold")
    plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("3. Watershed Boundaries")
    plt.axis('off')
    plt.tight_layout(); plt.show()