# utils.py
import numpy as np
import matplotlib.pyplot as plt
import os

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = y_true.squeeze() > 0.5  # Ensure binary
    y_pred = y_pred.squeeze() > 0.5
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def save_predictions(images, masks, preds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, mask, pred) in enumerate(zip(images, masks, preds)):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img.squeeze(), cmap='gray')
        ax[0].set_title("Image")
        ax[1].imshow(mask.squeeze(), cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(pred.squeeze(), cmap='gray')
        ax[2].set_title("Prediction")
        plt.savefig(os.path.join(save_dir, f"result_{i}.png"))
        plt.close()

