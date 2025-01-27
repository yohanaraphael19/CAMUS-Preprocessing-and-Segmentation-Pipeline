import torch
from tqdm import tqdm
from utils import dice_coefficient

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, pred, target):
        bce = torch.nn.functional.binary_cross_entropy(pred, target)
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return bce + dice_loss

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
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
        for images, masks in tqdm(val_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate Dice
            outputs_bin = (outputs > 0.5).float()
            dice = dice_coefficient(outputs_bin.cpu().numpy(), masks.cpu().numpy())
            
            running_loss += loss.item()
            running_dice += dice
    
    return running_loss / len(val_loader), running_dice / len(val_loader)
