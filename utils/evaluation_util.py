import torch
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


def tensor_to_numpy_img(tensor_img):
    """
    Expect tensor_img in CxHxW with values in [0,1]. Returns HxWxC in uint8 [0,255].
    """
    arr = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def calc_psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    A higher PSNR value indicates a higher-quality image reconstruction.

    :param img1: The first image (e.g., ground truth HR image).
    :param img2: The second image (e.g., predicted SR image).
    :return: The calculated PSNR value.
    """
    # Convert images to float64 to ensure precision during calculation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    return 10 * math.log10(255.0**2 / mse)


def calc_ssim(img1, img2, win_size=7):
    # Determine the largest odd win_size that fits the image
    h, w = img1.shape[:2]
    max_win = min(h, w)
    if max_win < win_size:
        # ensure odd number >= 3
        win_size = max(3, max_win if max_win % 2 == 1 else max_win - 1)

    return ssim_metric(
        img1, img2,
        data_range=255,
        channel_axis=-1,     # for color images (newer skimage)
        win_size=win_size
    )
