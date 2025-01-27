import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def get_image_and_mask_paths(image_dir, mask_dir):
    image_paths = []
    mask_paths = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):  # Only consider PNG files
                image_paths.append(os.path.join(root, file))
    
    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.endswith('.png'):  # Only consider PNG files
                mask_paths.append(os.path.join(root, file))
    
    image_paths.sort()
    mask_paths.sort()
    
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")
    return image_paths, mask_paths


def load_dataset(image_dir, mask_dir):
    image_paths, mask_paths = get_image_and_mask_paths(image_dir, mask_dir)
    
    assert len(image_paths) > 0, "No images found in the directory!"
    assert len(mask_paths) > 0, "No masks found in the directory!"
    assert len(image_paths) == len(mask_paths), "Mismatch between images and masks!"
    
    split_idx = int(0.8 * len(image_paths))
    train_images = image_paths[:split_idx]
    train_masks = mask_paths[:split_idx]
    val_images = image_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    
    return train_images, train_masks, val_images, val_masks


class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

        self.default_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure single channel
            transforms.Resize((256, 256)),               # Resize to 256x256
            transforms.ToTensor(),                       # Convert to Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load images and masks using PIL
        image = Image.open(self.images[idx]).convert("L")  # Convert to grayscale
        mask = Image.open(self.masks[idx]).convert("L")    # Convert to grayscale

        # Apply default transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = self.default_transform(image)
            mask = self.default_transform(mask)

        return image, mask


