import cv2
import numpy as np
import pytesseract


# Load the image using OpenCV
image = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Increase the contrast of the image using histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# Remove noise from the image using Gaussian filtering
gray = cv2.GaussianBlur(gray, (3,3), 0)

# Sharpen the image
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
gray = cv2.filter2D(gray, -1, kernel)

# Perform thresholding on the image to make the text more distinguishable
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Use Py-Tesseract to extract the text from the image
text = pytesseract.image_to_string(thresh, lang='eng')
print(text)
