import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select your image file in the popup window...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load image in grayscale (Fourier is usually done on single channel)
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (300, 300))

        # --- FOURIER TRANSFORM STEPS ---
        
        # 1. Perform the 2D Fast Fourier Transform
        dft = np.fft.fft2(img)
        
        # 2. Shift the zero-frequency component to the center of the spectrum
        # This makes it look like the VLab "Processed Image"
        dft_shift = np.fft.fftshift(dft)
        
        # 3. Calculate Magnitude Spectrum (Log scale to make it visible)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

        # 4. Inverse Fourier Transform (To show we can go back to pixels)
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # --- DISPLAY RESULTS ---
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Initial Image (Spatial)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Processed Image (Frequency)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(img_back, cmap='gray')
        plt.title('Reconstructed Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()