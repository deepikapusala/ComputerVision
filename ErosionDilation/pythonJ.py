import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image to see the Image Dilation and Erosion...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (400, 400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # --- PRE-PROCESSING ---
        # Convert to grayscale then Binary for Morphological Operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Define Structuring Element (Kernel)
        kernel = np.ones((5, 5), np.uint8)

        # --- MORPHOLOGICAL OPERATIONS ---
        # 1. Erosion: Shrinks the white objects
        eroded = cv2.erode(binary_img, kernel, iterations=1)

        # 2. Dilation: Expands the white objects
        dilated = cv2.dilate(binary_img, kernel, iterations=1)

        # --- DISPLAY RESULTS (4 Images) ---
        titles = ['1. Original Input', '2. Binary Version', 
                  '3. After Erosion', '4. After Dilation']
        images = [img_rgb, binary_img, eroded, dilated]

        plt.figure(figsize=(12, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            # Use gray colormap for B&W images, none for RGB
            if i == 0:
                plt.imshow(images[i])
            else:
                plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()