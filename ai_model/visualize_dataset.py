import cv2
import matplotlib.pyplot as plt

# Actual paths based on your screenshot
image_path = r"C:\Users\PC\Desktop\Train\Train\Urban\images_png\1367.png"
mask_path = r"C:\Users\PC\Desktop\Train\Train\Urban\masks_png\1367.png"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Read mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print("Error: Mask not found!")
    exit()

# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Urban Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Segmentation Mask")
plt.axis("off")

plt.tight_layout()
plt.show()