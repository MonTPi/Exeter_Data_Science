
import cv2
import numpy as np

# Load the image
img = cv2.imread("daisy.JPG")

cv2.imshow('test',img)

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of bright yellow colors in HSV color space
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Threshold the image to extract bright yellow flowers
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Apply morphological operations to remove small noise and fill in holes
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Apply the mask to the original image to extract the flowers
result = cv2.bitwise_and(img, img, mask=mask)

# Display the result
cv2.imshow("Result", result)

