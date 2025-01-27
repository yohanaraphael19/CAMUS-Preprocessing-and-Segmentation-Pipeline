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


class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, is_train=True):
        self.images = images
        self.masks = masks
        self.is_train = is_train
        
        # Image transformations (with augmentation for training)
        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5) if is_train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(degrees=10) if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
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
        
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # Ensure binary mask
        
        return image, mask

def load_dataset(image_dir, mask_dir, val_split=0.2):
    image_paths, mask_paths = get_image_and_mask_paths(image_dir, mask_dir)
    assert len(image_paths) > 0, "No matched image-mask pairs found!"
    
    split_idx = int((1 - val_split) * len(image_paths))
    return (
        (image_paths[:split_idx], mask_paths[:split_idx]),  # Train
        (image_paths[split_idx:], mask_paths[split_idx:])   # Val
    )


