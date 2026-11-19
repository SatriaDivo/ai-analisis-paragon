import cv2
import numpy as np

img = cv2.imread(r"testing/output/c0490_left_half.jpg")
h, w = img.shape[:2]

# In C0490:
# Glove is purple:
# Let's find contours of glove:
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Purple in HSV: H around 120-160
mask_purple = cv2.inRange(hsv, (110, 30, 100), (160, 255, 255))
cnts, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Purple contours found:", len(cnts))
for c in cnts:
    if cv2.contourArea(c) > 5000:
        x, y, cw, ch = cv2.boundingRect(c)
        print(f"Hand Bounding Box: x={x}, y={y}, w={cw}, h={ch}")
        # Top-right of hand is where the nozzle is
        print(f"Nozzle approximate position: x={x+cw}, y={y}")

