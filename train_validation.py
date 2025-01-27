import torch
from tqdm import tqdm

from utils import dice_coefficient

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        masks = (masks > 0.5).float()  # Convert to binary (0 or 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)

            masks = (masks > 0.5).float() 

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Compute dice coefficient
            outputs = (outputs > 0.5).float()
            dice = dice_coefficient(masks.cpu().numpy(), outputs.cpu().numpy())
            running_dice += dice
    
    avg_loss = running_loss / len(val_loader)
    avg_dice = running_dice / len(val_loader)
    return avg_loss, avg_dice
