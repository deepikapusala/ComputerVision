import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select that bird image for a PERFECT classification...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected!")
else:
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
    else:
        # Resize for speed but keep enough detail for the bird
        img = cv2.resize(img, (300, 300))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape to a list of pixels
        Z = img_rgb.reshape((-1, 3))
        Z = np.float32(Z)

        # --- THE CLASSIFICATION ENGINE (K-Means) ---
        # We define criteria: (type, max_iter, epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # K=2 means we want to classify into 2 groups: Bird vs Background
        K = 2 
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to uint8 and reshape to original image size
        center = np.uint8(center)
        res = center[label.flatten()]
        result_img = res.reshape((img_rgb.shape))

        # --- DISPLAY RESULTS ---
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title('1. Original Input')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        # This will show the image simplified into 2 main 'color patterns'
        plt.imshow(result_img)
        plt.title('2.Pattern Classification')
        plt.axis('off')

        plt.tight_layout()
        plt.show()