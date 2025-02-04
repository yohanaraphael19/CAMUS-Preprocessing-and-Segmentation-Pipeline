import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import EchoDataset, load_dataset
from model import UNet
from train_validation import train, validate, DiceBCELoss
from utils import save_predictions

# Config
config = {
    "image_dir": "/path/to/images", #please put your path to png images files
    "mask_dir": "/path/to/masks", #path to png masks files
    "batch_size": 8,
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-5,  # L2 Regularization
    "dropout_rate": 0.3,  # Dropout in UNet
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_metrics_dir": "./saved_metrics",
    "save_images_dir": "./saved_images"
}

# Initialize Model with Dropout
model = UNet(dropout_rate=config["dropout_rate"]).to(config["device"])

# Adam optimizer with L2 Regularization
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
criterion = DiceBCELoss()

# ReduceLROnPlateau for learning rate adjustment
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Handle paths with spaces using os.path
config["image_dir"] = os.path.expanduser(config["image_dir"])
config["mask_dir"] = os.path.expanduser(config["mask_dir"])
os.makedirs(config["save_metrics_dir"], exist_ok=True)
os.makedirs(config["save_images_dir"], exist_ok=True)

# Load Data with Augmentation
(train_images, train_masks), (val_images, val_masks) = load_dataset(
    config["image_dir"], config["mask_dir"], val_split=0.2
)

train_loader = DataLoader(
    EchoDataset(train_images, train_masks, is_train=True, augment=True),  # Enable Augmentation
    batch_size=config["batch_size"], shuffle=True
)

val_loader = DataLoader(
    EchoDataset(val_images, val_masks, is_train=False, augment=False),  # No augmentation for validation
    batch_size=config["batch_size"], shuffle=False
)

# Track Metrics
history = {
    "train_loss": [], "train_dice": [], "train_iou": [], "train_accuracy": [],
    "val_loss": [], "val_dice": [], "val_iou": [], "val_accuracy": []
}

# Training Loop with Early Stopping
best_dice = 0.0
patience = 10
wait = 0

for epoch in range(config["epochs"]):
    train_metrics = train(model, train_loader, optimizer, criterion, config["device"])
    val_metrics = validate(model, val_loader, criterion, config["device"])

    # Update History
    for key in ["loss", "dice", "iou", "accuracy"]:
        history[f"train_{key}"].append(train_metrics[key])
        history[f"val_{key}"].append(val_metrics[key])
    
    # Adjust Learning Rate
    scheduler.step(val_metrics["loss"])

    # Print Metrics
    print(f"\nEpoch {epoch+1}/{config['epochs']}")
    print(f"Train Loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

    # Save Best Model
    if val_metrics["dice"] > best_dice:
        best_dice = val_metrics["dice"]
        torch.save(model.state_dict(), os.path.join(config["save_metrics_dir"], "best_model.pth"))
        print(f"New best model saved with Val Dice: {best_dice:.4f}")
        wait = 0  # Reset patience counter
    else:
        wait += 1
        print(f"No improvement in Val Dice for {wait} epochs.")

    # Early Stopping
    if wait >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs. Best Val Dice: {best_dice:.4f}")
        break

# Save Training Metrics Plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history["train_dice"], label="Train Dice")
plt.plot(history["val_dice"], label="Val Dice")
plt.title("Dice Coefficient")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history["train_iou"], label="Train IoU")
plt.plot(history["val_iou"], label="Val IoU")
plt.title("IoU Score")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history["train_accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(config["save_metrics_dir"], "training_metrics.png"))
plt.close()

# Load Best Model for Predictions
model.load_state_dict(torch.load(os.path.join(config["save_metrics_dir"], "best_model.pth")))
model.eval()

# Get Sample Validation Data
sample_images, sample_masks = next(iter(val_loader))
sample_images = sample_images.to(config["device"])

with torch.no_grad():
    sample_preds = model(sample_images).cpu().numpy()

# Denormalize Function
def denormalize(image):
    return image * 0.5 + 0.5  # Reverse Normalize(mean=0.5, std=0.5)

sample_images = denormalize(sample_images.cpu().numpy())
sample_masks = sample_masks.numpy()

# Save Predictions
save_predictions(
    images=sample_images,
    masks=sample_masks,
    preds=sample_preds,
    save_dir=config["save_images_dir"]
)

print(f"\nTraining metrics saved to: {config['save_metrics_dir']}")
print(f"Prediction images saved to: {config['save_images_dir']}")
