import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1. Open local file picker
root = tk.Tk()
root.withdraw()
print("Select a COLORED image for Hough Transform...")
image_path = filedialog.askopenfilename(title="Select Image", 
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

if not image_path:
    print("No file selected!")
else:
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
    else:
        img = cv2.resize(img, (500, 500))
        img_lines = img.copy()
        img_circles = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- LINE DETECTION ---
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- CIRCLE DETECTION ---
        # Param1 and Param2 control sensitivity; 50 is a safe middle ground
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=50, param2=50, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img_circles, (i[0], i[1]), i[2], (255, 0, 0), 2) # Blue circle
                cv2.circle(img_circles, (i[0], i[1]), 2, (0, 0, 255), 3)    # Red center

        # --- DISPLAY ---
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("1. Original")
        plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)); plt.title("2. Lines Detected")
        plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB)); plt.title("3. Circles Detected")
        plt.axis('off')
        plt.tight_layout(); plt.show()