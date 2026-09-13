import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import UrbanDataset
from model import model


# -----------------------------
# DEVICE
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# DATASET PATHS
# -----------------------------

IMAGE_DIR = r"C:\Users\PC\Desktop\Train\Train\Urban\images_png"

MASK_DIR = r"C:\Users\PC\Desktop\Train\Train\Urban\masks_png"


# -----------------------------
# LOAD DATASET
# -----------------------------

dataset = UrbanDataset(
    IMAGE_DIR,
    MASK_DIR
)

print("Total dataset:", len(dataset))


# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# DATALOADERS
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# MODEL
# -----------------------------

model = model.to(device)


# -----------------------------
# LOSS FUNCTION
# -----------------------------

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# -----------------------------
# OPTIMIZER
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# TRAINING
# -----------------------------

EPOCHS = 10

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_unet.pth"
        )

        print("Best model saved!")


print("Training completed!")