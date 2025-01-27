# utils.py
import numpy as np
import matplotlib.pyplot as plt
import os

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def save_predictions(images, predictions, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (image, pred) in enumerate(zip(images, predictions)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(image.squeeze(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.savefig(os.path.join(save_dir, f"result_{i}.png"))
        plt.close()

