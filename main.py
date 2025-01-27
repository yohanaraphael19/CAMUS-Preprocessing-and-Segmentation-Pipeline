import os
import torch
from torch.utils.data import DataLoader
from dataset import EchoDataset, load_dataset
from model import UNet
from train_validation import train, validate
from utils import save_predictions

# Paths
image_dir = "C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/NIFTI TO PNG/output_data/good-output/images"
mask_dir = "C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/NIFTI TO PNG/output_data/good-output/masks"
save_dir = "C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/CAMUS-saved-images"
checkpoint_dir = "C:/Users/Blessedsoul/PycharmProjects/Segmentation 1/checkpoints"

# Parameters
batch_size = 8
epochs = 20
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs(save_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Data
train_images, train_masks, val_images, val_masks = load_dataset(image_dir, mask_dir)
train_dataset = EchoDataset(train_images, train_masks)
val_dataset = EchoDataset(val_images, val_masks)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = UNet().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with tqdm and checkpoints
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_dice = validate(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
    
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))

# Save predictions
images, _ = next(iter(val_loader))
predictions = model(images.to(device)).detach().cpu().numpy()
save_predictions(images.numpy(), predictions, save_dir)
