import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Use relative imports since this is a module within the 'utils' package
from .image_util import tensor_to_numpy, save_tensor_img, show_tensor_img, make_lr_from_hr
from .image_util import crop, random_flip, random_rotate
from .evaluation_util import calc_psnr, calc_ssim
from utils.dataset import SRDataset

# Assuming the EDSR model is in a separate 'model' directory as per your structure.

from model.edsr import EDSR


def save_model(model, optimizer, epoch, path):
    """
    Saves the model and optimizer states to a specified path.

    This function is useful for saving training progress and resuming training
    from a specific checkpoint. The saved file includes the epoch number,
    the model's state dictionary, and the optimizer's state dictionary.

    :param model: The EDSR model to save.
    :param optimizer: The optimizer used for training.
    :param epoch: The current epoch number.
    :param path: The directory path where the checkpoint will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    checkpoint_path = os.path.join(path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


def load_model(model, optimizer, path):
    """
    Loads a model and optimizer state from a specified checkpoint path.

    This function is used to resume training or load a pre-trained model
    for evaluation or inference. It handles the case where no checkpoint is found.

    :param model: The EDSR model instance.
    :param optimizer: The optimizer instance.
    :param path: The full path to the checkpoint file to be loaded.
    :return: The epoch number from which training can be resumed.
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}. Starting from epoch 0.")
        return 0
    
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming training from epoch {epoch + 1}.")
    return epoch


def inference(model, lr_image_tensor):
    """
    Performs inference on a low-resolution image using the trained model.

    :param model: The trained EDSR model.
    :param lr_image_tensor: A low-resolution image as a PyTorch tensor.
                            The tensor should be in the format CxHxW.
    :return: The super-resolved high-resolution image as a PyTorch tensor.
    """
    model.eval()
    with torch.no_grad():
        sr_image_tensor = model(lr_image_tensor.unsqueeze(0))
    return sr_image_tensor.squeeze(0)
