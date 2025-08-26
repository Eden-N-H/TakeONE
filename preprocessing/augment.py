import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import image_util as util

if __name__ == '__main__':
    
    util.augment_dir(train_root='./dataset/Raw Data/low_res',
                     target_root='./dataset/Raw Data/high_res',
                     train_output_path='./dataset/train/low_res',
                     target_output_path='./dataset/train/high_res',
                     aug_num=20, hr_crop_size=192, scale=4)