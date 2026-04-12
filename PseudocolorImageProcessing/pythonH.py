import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
# As per your request, using the colored image prompt, 
# but remember: the effect is best seen on grayscale data!
print("Select a COLORED image to see the Pseudocolor Processing...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load image and force it to grayscale to demonstrate pseudocoloring
    img = cv2.imread(image_path, 0) 
    
    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (400, 400))

        # --- PSEUDOCOLOR MAPPING ---
        # We take the 1-channel grayscale and map it to 3-channel color

        # 1. Jet Colormap (Classic 'Thermal' look)
        pseudocolor_jet = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        # 2. Hot Colormap (Black -> Red -> Yellow -> White)
        pseudocolor_hot = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

        # 3. Ocean Colormap (Shades of blue and green)
        pseudocolor_ocean = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

        # Convert BGR (OpenCV default) to RGB for Matplotlib display
        pseudocolor_jet = cv2.cvtColor(pseudocolor_jet, cv2.COLOR_BGR2RGB)
        pseudocolor_hot = cv2.cvtColor(pseudocolor_hot, cv2.COLOR_BGR2RGB)
        pseudocolor_ocean = cv2.cvtColor(pseudocolor_ocean, cv2.COLOR_BGR2RGB)

        # --- DISPLAY RESULTS ---
        titles = ['Original Grayscale', '1. Jet (Thermal)', 
                  '2. Hot (Intensity)', '3. Ocean (Depth)']
        images = [img, pseudocolor_jet, pseudocolor_hot, pseudocolor_ocean]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            if i == 0:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()