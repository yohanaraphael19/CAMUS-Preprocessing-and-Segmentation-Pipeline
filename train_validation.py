import torch
from tqdm import tqdm
from utils import dice_coefficient, iou_score, accuracy
import torch.nn as nn
import numpy as np

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

def train(model, train_loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    dice_scores, iou_scores, accuracies = [], [], []

    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = (outputs > 0.5).float().cpu().numpy()
        masks_np = masks.cpu().numpy()

        dice_scores.append(dice_coefficient(masks_np, preds))
        iou_scores.append(iou_score(masks_np, preds))
        accuracies.append(accuracy(masks_np, preds))

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_acc = np.mean(accuracies)

    if scheduler:
        scheduler.step(avg_loss)  # Adjust learning rate based on validation loss

    return {"loss": avg_loss, "dice": avg_dice, "iou": avg_iou, "accuracy": avg_acc}

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores, iou_scores, accuracies = [], [], []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = (outputs > 0.5).float().cpu().numpy()
            masks_np = masks.cpu().numpy()

            dice_scores.append(dice_coefficient(masks_np, preds))
            iou_scores.append(iou_score(masks_np, preds))
            accuracies.append(accuracy(masks_np, preds))

            running_loss += loss.item()

    avg_loss = running_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_acc = np.mean(accuracies)

    return {"loss": avg_loss, "dice": avg_dice, "iou": avg_iou, "accuracy": avg_acc}

