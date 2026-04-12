import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image for Blind Deconvolution (OpenCV Version)...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected!")
else:
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
    else:
        img = cv2.resize(img, (400, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to gray for mathematical restoration
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- STEP 1: CREATE BLUR (The 'Degradation') ---
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        blurred = cv2.filter2D(gray, -1, kernel)

        # --- STEP 2: PSEUDO-BLIND DECONVOLUTION (Wiener Filter Approach) ---
        # We use the FFT (Fast Fourier Transform) to reverse the blur
        # without needing a complex iterative library.
        dummy = np.copy(blurred)
        psf = np.ones((5, 5)) / 25.0
        
        # Pad PSF to match image size
        psf_padded = np.zeros_like(gray, dtype=np.float32)
        psf_padded[:5, :5] = psf
        
        # Fourier Transforms
        IMG = np.fft.fft2(blurred)
        PSF = np.fft.fft2(psf_padded)
        
        # Deconvolution formula: G / (H + noise_ratio)
        # We assume a small noise ratio (0.01) since we don't know the exact noise
        noise_ratio = 0.01
        RESTORED = IMG / (PSF + noise_ratio)
        
        # Inverse Fourier to get image back
        deconvolved = np.abs(np.fft.ifft2(RESTORED))
        deconvolved = np.clip(deconvolved, 0, 255).astype(np.uint8)

        # --- DISPLAY RESULTS (3 Images) ---
        titles = ['1. Original Input', '2. Blurred (Degraded)', '3. Restored (Deconvolved)']
        images = [img_rgb, blurred, deconvolved]

        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            if i == 0:
                plt.imshow(images[i])
            else:
                plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()