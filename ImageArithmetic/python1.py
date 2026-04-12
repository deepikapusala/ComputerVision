import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# 1. Open a window to pick the file instead of typing
root = tk.Tk()
root.withdraw() # Hide the tiny extra window
image_path = filedialog.askopenfilename(title="Select your cricket.png file", 
                                        filetypes=[("Image files", "*.png *.jpeg")])

if not image_path:
    print("No file selected!")
else:
    # Load the image
    img = cv2.imread(image_path, 0)
    
    if img is None:
        print("Error: Could not load image. Check the file format.")
    else:
        img = cv2.resize(img, (500, 500)) 

        # A. Create Noise
        noise = np.random.normal(0, 25, img.shape).astype('int16')
        noisy_image = np.clip(img.astype('int16') + noise, 0, 255).astype('uint8')

        # B. Arithmetic Operation: SUBTRACTION
        diff_image = cv2.absdiff(img, noisy_image)

        # C. Arithmetic Operation: MULTIPLICATION
        bright_image = np.clip(img.astype('float32') * 1.5, 0, 255).astype('uint8')

        # --- DISPLAY RESULTS ---
        titles = ['Original Input', 'Corrupted (Additive Noise)', 
                  'Subtraction (Noise Map)', 'Multiplication (Gain)']
        images = [img, noisy_image, diff_image, bright_image]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()