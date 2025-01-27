import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import EchoDataset, load_dataset
from model import UNet
from train_validation import train, validate, DiceBCELoss
from utils import save_predictions

# Config
config = {
    "image_dir": "/path/to/images", #please put the path to PNG images
    "mask_dir": "/path/to/masks", #please put the path to PNG masks
    "batch_size": 8,
    "epochs": 50,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./saved_predictions"  # Add save directory for the predicted output
}

# Initialize
model = UNet().to(config["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
criterion = DiceBCELoss()

# Data
(train_images, train_masks), (val_images, val_masks) = load_dataset(
    config["image_dir"], config["mask_dir"], val_split=0.2
)

train_dataset = EchoDataset(train_images, train_masks, is_train=True)
val_dataset = EchoDataset(val_images, val_masks, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Training loop
best_dice = 0.0
for epoch in range(config["epochs"]):
    train_loss = train(model, train_loader, optimizer, criterion, config["device"])
    val_loss, val_dice = validate(model, val_loader, criterion, config["device"])
    
    print(f"Epoch {epoch+1}/{config['epochs']}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
    
    # Save best model
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "best_model.pth")

# --------------------------
# Save predictions after training
# --------------------------
def denormalize(image):
    # Reverse Normalize (mean=0.5, std=0.5)
    return image * 0.5 + 0.5

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Get a batch of validation data
val_images, val_masks = next(iter(val_loader))
val_images = val_images.to(config["device"])
with torch.no_grad():
    val_preds = model(val_images).cpu().numpy()

# Denormalize images and convert to numpy
val_images = denormalize(val_images.cpu().numpy())
val_masks = val_masks.cpu().numpy()

# Save predictions
save_predictions(
    images=val_images,
    masks=val_masks,
    preds=val_preds,
    save_dir=config["save_dir"]
)

print(f"Predictions saved to {config['save_dir']}!")
