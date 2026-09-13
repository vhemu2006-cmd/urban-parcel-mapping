import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "best_unet.pth"

IMAGE_PATH = r"C:\Users\PC\Desktop\Train\Train\Urban\images_png\1368.png"

OUTPUT_PATH = "colored_prediction.png"

# -----------------------------
# Class colors
# -----------------------------
# Class IDs:
# 0 = No-data
# 1 = Background
# 2 = Building
# 3 = Road
# 4 = Water
# 5 = Barren land
# 6 = Forest
# 7 = Agriculture

CLASS_NAMES = {
    0: "No-data",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren Land",
    6: "Forest",
    7: "Agriculture"
}

# RGB colors
CLASS_COLORS = {
    0: [0, 0, 0],          # Black
    1: [255, 255, 255],    # White
    2: [255, 0, 0],        # Red
    3: [255, 255, 0],      # Yellow
    4: [0, 0, 255],        # Blue
    5: [165, 42, 42],      # Brown
    6: [0, 255, 0],        # Green
    7: [144, 238, 144]     # Light green
}

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load model
# -----------------------------
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=8
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

print("Trained model loaded successfully!")

# -----------------------------
# Load image
# -----------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError("Image not found. Check IMAGE_PATH.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_resized = cv2.resize(
    image,
    (256, 256),
    interpolation=cv2.INTER_LINEAR
)

# -----------------------------
# Prepare tensor
# -----------------------------
input_image = image_resized.astype(np.float32) / 255.0

input_tensor = torch.tensor(input_image)
input_tensor = input_tensor.permute(2, 0, 1)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)

prediction = prediction.squeeze().cpu().numpy()

print("Prediction completed!")
print("Predicted classes:", np.unique(prediction))

# -----------------------------
# Create colored mask
# -----------------------------
colored_mask = np.zeros(
    (prediction.shape[0], prediction.shape[1], 3),
    dtype=np.uint8
)

for class_id, color in CLASS_COLORS.items():
    colored_mask[prediction == class_id] = color

# -----------------------------
# Calculate class percentages
# -----------------------------
total_pixels = prediction.size

print("\nLand-use analysis:")

for class_id, class_name in CLASS_NAMES.items():
    pixel_count = np.sum(prediction == class_id)
    percentage = (pixel_count / total_pixels) * 100

    if pixel_count > 0:
        print(
            f"{class_name}: "
            f"{pixel_count} pixels "
            f"({percentage:.2f}%)"
        )

# -----------------------------
# Display results
# -----------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_resized)
plt.title("Original Aerial Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(colored_mask)
plt.title("Colored Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_resized)
plt.imshow(colored_mask, alpha=0.5)
plt.title("Overlay Result")
plt.axis("off")

# -----------------------------
# Add legend
# -----------------------------
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=np.array(CLASS_COLORS[1]) / 255, label="Background"),
    Patch(facecolor=np.array(CLASS_COLORS[2]) / 255, label="Building"),
    Patch(facecolor=np.array(CLASS_COLORS[3]) / 255, label="Road"),
    Patch(facecolor=np.array(CLASS_COLORS[4]) / 255, label="Water"),
    Patch(facecolor=np.array(CLASS_COLORS[5]) / 255, label="Barren Land"),
    Patch(facecolor=np.array(CLASS_COLORS[6]) / 255, label="Forest"),
    Patch(facecolor=np.array(CLASS_COLORS[7]) / 255, label="Agriculture"),
]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_resized)
plt.title("Original Aerial Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(colored_mask)
plt.title("Colored Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_resized)
plt.imshow(colored_mask, alpha=0.5)
plt.title("Overlay Result")
plt.axis("off")

plt.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(-0.7, -0.08),
    ncol=4
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches="tight")

print("\nColored output saved as:", OUTPUT_PATH)

print("\nColored output saved as:", OUTPUT_PATH)