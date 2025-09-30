import os
import argparse
import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from model.edsr import EDSR
from utils.evaluation_util import tensor_to_numpy_img

def load_model(checkpoint_path, scale=4, device="cuda"):
    model = EDSR(scale=scale)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def run_inference(model, lr_image_path, device="cuda"):
    img = cv.cvtColor(cv.imread(lr_image_path), cv.COLOR_BGR2RGB)  # BGR
    if img is None:
        raise FileNotFoundError(f"Could not read {lr_image_path}")
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(tensor)
    sr_img = tensor_to_numpy_img(sr.squeeze(0))  # returns H×W×C, RGB 0–255
    return sr_img

def save_image(image_np, out_path):
    # Convert RGB→BGR for OpenCV save
    cv.imwrite(out_path, cv.cvtColor(image_np, cv.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EDSR checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pth/.pt file")
    parser.add_argument("--lr_image", type=str, required=True,
                        help="Path to low-resolution image to upscale")
    parser.add_argument("--output", type=str, default="sr_result.png",
                        help="Where to save the SR output")
    parser.add_argument("--scale", type=int, default=4,
                        help="Upscale factor (default 4)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.checkpoint} on {device}...")
    model = load_model(args.checkpoint, scale=args.scale, device=device)

    print(f"Upscaling {args.lr_image} ...")
    sr_img = run_inference(model, args.lr_image, device=device)

    save_image(sr_img, args.output)
    print(f"✅ Saved super-resolved image to {args.output}")
