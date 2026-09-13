import torch
import segmentation_models_pytorch as smp


# Number of segmentation classes
NUM_CLASSES = 8


# Create U-Net model
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
)


# Print model
print(model)


# Test with a dummy image
dummy_input = torch.randn(1, 3, 256, 256)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)