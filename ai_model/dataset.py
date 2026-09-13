import os
import cv2
import torch
from torch.utils.data import Dataset


class UrbanDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

        self.masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith(".png")
        ])

        assert len(self.images) == len(self.masks), (
            f"Images: {len(self.images)}, Masks: {len(self.masks)}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = os.path.join(
            self.image_dir,
            self.images[index]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[index]
        )

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Read mask
        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise FileNotFoundError(mask_path)

        # Resize
        image = cv2.resize(
            image,
            (256, 256)
        )

        mask = cv2.resize(
            mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert image to tensor
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        image = image.permute(2, 0, 1) / 255.0

        # Convert mask to tensor
        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask