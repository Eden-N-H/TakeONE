import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import Dataset
from torchvision.transforms import ToTensor
from utils.trainer import train_model

if __name__ == '__main__':
    # load dataset
    print('loading dataset ...')

    train_data = Dataset(
        train_root="./dataset/train/low_res",
        target_root="./dataset/train/high_res",
        transform=ToTensor()
    )

    val_data = Dataset(
        train_root="./dataset/val/low_res",
        target_root="./dataset/val/high_res",
        transform=ToTensor()
    )

    # train the model
    train_model(
        scale=4,
        train_dataset=train_data,
        epoch=100,
        lr=1e-4,
        batch_size=16,
        checkpoint_save_path='checkpoints',
        checkpoint=False,
        checkpoint_load_path=None,
        cuda=False
    )