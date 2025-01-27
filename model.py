#project structure
#├── dataset.py            Dataset and preprocessing for PNG files
#├── model.py              U-Net architecture
#├── utils.py              Helper functions (e.g., metrics, visualizations)
#├── train_validation.py   Training and validation loop
#├── main.py               Main script to coordinate everything
#├── saved_images/         Folder to save segmentation results


# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.contracting_block(1, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.bottleneck = self.contracting_block(128, 256)
        self.dec2 = self.expansive_block(256, 128)
        self.dec1 = self.expansive_block(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def expansive_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        bottleneck = self.bottleneck(enc2)
        dec2 = self.dec2(bottleneck)
        dec1 = self.dec1(dec2 + enc2)
        out = self.final(dec1 + enc1)

        # Resize output to match the size of the masks (256x256)
        out_resized = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        return torch.sigmoid(out_resized)

# In your train function, you can leave the rest as is, but the resizing will ensure the output and masks are compatible in size.



