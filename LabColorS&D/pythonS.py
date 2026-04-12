import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk(); root.withdraw()
print("Select a COLORED image for a PERFECT Delta E map...")
path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not path:
    print("No file selected!")
else:
    img = cv2.imread(path)
    img = cv2.resize(img, (400, 400))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Convert to LAB Space (using float32 for precision)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)

    # 3. Create a Modified version (Shift colors so there IS a difference)
    modified_lab = img_lab.copy()
    modified_lab[:,:,1] += 30  # Shift Red/Green
    modified_lab[:,:,2] -= 20  # Shift Blue/Yellow
    
    # Reconstruct RGB for display
    modified_rgb = cv2.cvtColor(np.clip(modified_lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2RGB)

    # 4. Calculate Delta E (Euclidean Distance in LAB space)
    diff = img_lab - modified_lab
    delta_e = np.sqrt(np.sum(diff**2, axis=2))

    # --- THE FIX: NORMALIZATION & COLORMAP ---
    # This ensures the map is NEVER black
    delta_e_norm = cv2.normalize(delta_e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply a Jet or Magma colormap so the differences "glow"
    delta_e_colored = cv2.applyColorMap(delta_e_norm, cv2.COLORMAP_JET)
    delta_e_colored_rgb = cv2.cvtColor(delta_e_colored, cv2.COLOR_BGR2RGB)

    # --- DISPLAY (3 Flawless Images) ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(modified_rgb)
    plt.title("2. Modified (Color Shift)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(delta_e_colored_rgb)
    plt.title("3. Delta E (Visualized Map)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()