import torch.nn as nn

def minmax(img):
    img = (img- img.min())/(img.max()-img.min())
    return img  

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, pooling=None):
        super().__init__()
        self.pooling = pooling
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
    def forward(self, x):
        x = self.encoder(x)
        if self.pooling == 'max':
            x = nn.functional.max_pool2d(x, kernel_size=5)
        if self.pooling == 'mean':
            x = nn.functional.avg_pool2d(x, kernel_size=5)
        return x


