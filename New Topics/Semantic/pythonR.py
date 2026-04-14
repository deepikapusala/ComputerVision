import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image (with multiple objects) for Segmentation...")
image_path = filedialog.askopenfilename(title="Select Image")

if image_path:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 400))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- SEMANTIC SEGMENTATION (Grouping by 'Class') ---
    Z = img_rgb.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    semantic_res = center[label.flatten()].reshape((img_rgb.shape)).astype(np.uint8)

    # --- INSTANCE SEGMENTATION (Identifying Separate Objects) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Finding connected components (each object gets a unique ID/Color)
    ret, labels = cv2.connectedComponents(thresh)
    # Map labels to a color map so they are visible
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    instance_res = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

    # --- DISPLAY RESULTS ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(semantic_res)
    plt.title("2. Semantic (By Category)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(instance_res)
    plt.title("3. Instance (By Individual)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()