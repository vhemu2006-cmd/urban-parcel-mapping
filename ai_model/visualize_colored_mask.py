import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\PC\Desktop\Train\Train\Urban\images_png\1367.png"
mask_path = r"C:\Users\PC\Desktop\Train\Train\Urban\masks_png\1367.png"

print("Loading image...")

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

if mask is None:
    print("Mask not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create colored mask
colored_mask = np.zeros(
    (mask.shape[0], mask.shape[1], 3),
    dtype=np.uint8
)

colored_mask[mask == 0] = [0, 0, 0]
colored_mask[mask == 1] = [255, 255, 255]
colored_mask[mask == 2] = [255, 0, 0]
colored_mask[mask == 3] = [0, 0, 255]
colored_mask[mask == 4] = [0, 255, 255]
colored_mask[mask == 5] = [139, 69, 19]
colored_mask[mask == 6] = [0, 255, 0]
colored_mask[mask == 7] = [255, 255, 0]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Raw Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(colored_mask)
plt.title("Colored Segmentation Mask")
plt.axis("off")

plt.tight_layout()

# Save output image
plt.savefig("colored_mask_output.png")

print("Visualization saved as colored_mask_output.png")

plt.show()