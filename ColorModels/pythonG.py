import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image to see the Color Models...")
image_path = filedialog.askopenfilename(title="Select Colored Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected! Exiting...")
else:
    # Load colored image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
    else:
        img = cv2.resize(img, (300, 300))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- 1. RGB (Red, Green, Blue) ---
        # We'll just show the original RGB reconstruction
        rgb_display = img_rgb

        # --- 2. HSL (Hue, Saturation, Lightness) ---
        img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # --- 3. CIELAB (L*a*b*) ---
        # L = Lightness, a = green-red, b = blue-yellow
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # --- 4. CMYK (Cyan, Magenta, Yellow, Black) ---
        # Mathematical conversion for the Cyan channel
        img_float = img_rgb.astype(float) / 255.0
        K = 1 - np.max(img_float, axis=2)
        C = (1 - img_float[..., 0] - K) / (1 - K + 1e-10)
        # We'll display the Cyan intensity map
        cyan_channel = np.clip(C * 255, 0, 255).astype(np.uint8)

        # --- DISPLAY RESULTS ---
        titles = ['1. RGB Model (Original)', '2. HSL Model (Gradients)', 
                  '3. CIELAB Model', '4. CMYK (Cyan Channel)']
        images = [rgb_display, img_hsl, img_lab, cyan_channel]

        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            # CMYK/Cyan is single channel, so we use a specific colormap
            if i == 3:
                plt.imshow(images[i], cmap='GnBu')
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()