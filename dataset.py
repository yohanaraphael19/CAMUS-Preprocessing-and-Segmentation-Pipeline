import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


class EchoDataset(Dataset):
    def __init__(self, images, masks, is_train=True, augment=False):
        self.images = images
        self.masks = masks
        self.is_train = is_train
        self.augment = augment  # Store the augmentation flag

        # Base transformations (applied to both training and validation)
        self.base_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Augmentations (only applied when training and augment=True)
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]) if self.augment else None

        # Mask transformations (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        mask = Image.open(self.masks[idx]).convert("L")

        # Apply augmentations if enabled
        if self.augment and self.augmentation_transforms:
            image = self.augmentation_transforms(image)

        image = self.base_transforms(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # Ensure binary mask

        return image, mask

def load_dataset(image_dir, mask_dir, val_split=0.2, shuffle=True):
    image_paths, mask_paths = get_image_and_mask_paths(image_dir, mask_dir)

    # Shuffle before splitting if enabled
    if shuffle:
        combined = list(zip(image_paths, mask_paths))
        random.shuffle(combined)
        image_paths, mask_paths = zip(*combined)

    split_idx = int((1 - val_split) * len(image_paths))

    return (
        (image_paths[:split_idx], mask_paths[:split_idx]),  # Train
        (image_paths[split_idx:], mask_paths[split_idx:])   # Val
    )
