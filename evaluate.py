import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_util import enhance


if __name__ == '__main__':
    # make sure you change the image_path to the path of the input image
    enhance(scale=4, image_path='dataset/Testing/low_res/duck.png', pre_train=True, weight_path=None, display=True,
            save=True, output_path='output/enhanced_image.png', cuda=False)