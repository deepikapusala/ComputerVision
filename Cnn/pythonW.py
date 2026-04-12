import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

root = tk.Tk(); root.withdraw()
print("Select a COLORED image for CNN Feature Extraction...")
path = filedialog.askopenfilename()

if path:
    img = cv2.imread(path)
    img = cv2.resize(img, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- SIMULATING A CNN LAYER ---
    # 1. Convolution (Edge Detection Filter)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    feature_map = cv2.filter2D(gray, -1, kernel)
    
    # 2. ReLU Activation (Remove negative values / darken low responses)
    relu_output = np.maximum(0, feature_map)

    # --- DISPLAY ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("1. Original Input")
    plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(feature_map, cmap='gray'); plt.title("2. Conv Layer (Features)")
    plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(relu_output, cmap='hot'); plt.title("3. ReLU Activation")
    plt.axis('off')
    plt.tight_layout(); plt.show()