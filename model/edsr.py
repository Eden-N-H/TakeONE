import math
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Standard EDSR residual block without batch normalization.
    Uses two 3×3 conv layers with ReLU in between and a residual scaling factor.
    """
    def __init__(self, num_channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                               padding=1, bias=True)

        # Kaiming initialization for stability
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual * self.res_scale


class Upsampler(nn.Sequential):
    """
    Upsampler using sub-pixel convolution (PixelShuffle).
    Supports scale factors 2^n (e.g., 2, 4, 8).
    """
    def __init__(self, scale, n_feat):
        m = []
        if (scale & (scale - 1)) == 0:      # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feat, 4 * n_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(n_feat, 9 * n_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
            m.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*m)


class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution (EDSR) model.

    Args:
        scale (int): Upscaling factor (e.g., 4 for x4 SR).
        num_channels (int): Number of feature maps in each conv layer.
        num_residual_blocks (int): Number of residual blocks.
        in_channels (int): Number of input image channels (default 3 for RGB).
        out_channels (int): Number of output image channels (default 3 for RGB).
    """
    def __init__(self, scale=4, num_channels=256, num_residual_blocks=32,
                 in_channels=3, out_channels=3):
        super().__init__()

        # Head: first conv
        self.head = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1,
                              padding=1, bias=True)

        # Body: N residual blocks + one conv
        body = [ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        body.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                              padding=1, bias=True))
        self.body = nn.Sequential(*body)

        # Upsampling
        self.upsample = Upsampler(scale, num_channels)

        # Tail: final conv
        self.tail = nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, bias=True)

        # Init head/tail
        nn.init.kaiming_normal_(self.head.weight, nonlinearity='relu')
        nn.init.zeros_(self.head.bias)
        nn.init.kaiming_normal_(self.tail.weight, nonlinearity='relu')
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        # x expected range [0,1]
        x = self.head(x)
        res = self.body(x)
        x = x + res  # global skip connection
        x = self.upsample(x)
        x = self.tail(x)
        return x
