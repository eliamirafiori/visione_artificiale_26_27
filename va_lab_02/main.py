import cv2
import numpy as np

# Create a sample synthetic image if one isn't on disk (gradient + shapes)
# Image dimensions: 300 height, 400 width, 3 channels (BGR)
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:, :] = [180, 105, 255]  # Background: Pink in BGR
cv2.circle(img, (200, 150), 60, (0, 255, 0), -1)  # Green circle
cv2.rectangle(img, (50, 50), (120, 120), (255, 0, 0), -1)  # Blue square
cv2.imwrite("sample_input.jpg", img)

# 1. Loading and Properties
# OpenCV reads images as NumPy arrays in BGR format (not RGB)
image = cv2.imread("sample_input.jpg")
if image is None:
    raise FileNotFoundError("Image could not be loaded.")

height, width, channels = image.shape
print(f"Dimensions: {width}x{height}, Channels: {channels}, Data type: {image.dtype}")

# 2. Color Space Conversions
# BGR to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BGR to HSV (Hue, Saturation, Value) - standard for color-based segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# BGR to RGB (required when displaying with matplotlib)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Basic Geometric Transformations
# Resizing (width, height)
resized = cv2.resize(image, (200, 150), interpolation=cv2.INTER_LINEAR)

# Cropping (uses NumPy array slicing: [y_start:y_end, x_start:x_end])
cropped = image[50:250, 100:300]

# Rotation around center
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

# Flipping (1 = horizontal, 0 = vertical, -1 = both)
flipped_horizontal = cv2.flip(image, 1)

# 4. Color Alterations & Arithmetic
# Adjusting brightness and contrast: new_img = alpha * img + beta
# alpha > 1.0 increases contrast; beta > 0 increases brightness
bright_contrast = cv2.convertScaleAbs(image, alpha=1.2, beta=40)

# Splitting and merging channels
b, g, r = cv2.split(image)
# Boost the Red channel and re-merge
r_boosted = cv2.add(r, 50)
merged = cv2.merge([b, g, r_boosted])

# 5. Filtering and Thresholding
# Gaussian Blur to smooth noise
blurred = cv2.GaussianBlur(image, (7, 7), sigmaX=1.5)

# Binary thresholding on grayscale image
# Pixels above 127 become 255 (white), others become 0 (black)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 6. Saving Outputs
cv2.imwrite("output_gray.jpg", gray)
cv2.imwrite("output_rotated.jpg", rotated)
cv2.imwrite("output_thresh.png", thresh)
print("Saved transformed images to disk.")

# 7. GUI Display (press any key to close the window)
cv2.imshow("Original (BGR)", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Threshold", thresh)

print("Press any key in an image window to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()
