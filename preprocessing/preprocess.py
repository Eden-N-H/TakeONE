import argparse
import os
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.image_util import make_lr_from_hr


def make_lr_from_hr(hr_path, lr_path, scale):
    from PIL import Image
    hr_image = Image.open(hr_path)
    w, h = hr_image.size
    lr_image = hr_image.resize((w // scale, h // scale), Image.BICUBIC)
    lr_image.save(lr_path)


def main():
    """
    Main function to parse arguments and create low-resolution images from a directory
    of high-resolution images.
    """
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description='Image Preprocessing for Super-Resolution')
    
    # Add arguments for input and output directories and the scale factor
    parser.add_argument('--hr_dir', type=str, required=True, 
                        help='Path to the directory containing high-resolution images')
    parser.add_argument('--lr_dir', type=str, required=True, 
                        help='Path to the directory where low-resolution images will be saved')
    parser.add_argument('--scale', type=int, default=2, 
                        help='The super-resolution scale factor')
    
    args = parser.parse_args()

    # Create the output directory if it doesn't already exist
    if not os.path.exists(args.lr_dir):
        os.makedirs(args.lr_dir)
        print(f"Created output directory: {args.lr_dir}")

    print(f"Starting to generate low-resolution images (scale={args.scale})...")
    
    # Iterate through all files in the high-resolution directory
    for filename in os.listdir(args.hr_dir):
        # Check if the file is a supported image format
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            hr_path = os.path.join(args.hr_dir, filename)
            lr_path = os.path.join(args.lr_dir, filename)
            
            # Open the high-resolution image
            hr_image = Image.open(hr_path)
            
            # Generate the low-resolution image using the function from image_util
            make_lr_from_hr(hr_path, lr_path, args.scale)
            print(f"Generated and saved: {lr_path}")

    print("\nPreprocessing complete. Low-resolution images are ready for training.")


if __name__ == '__main__':
    main()
