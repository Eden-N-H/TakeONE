import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2 as cv
from torchvision import transforms

from model.edsr import EDSR
from utils.evaluation_util import calc_psnr, calc_ssim, tensor_to_numpy_img

# --- SETTINGS ---
scale = 4
checkpoint_dir = "./checkpoints"
val_dir = ".\\dataset\DIV2K_valid_LRbicubx4"
hr_val_dir = ".\\dataset\\DIV2K_valid_HR"
cuda = torch.cuda.is_available()

NUM_VALIDATION_IMAGES = 20
START_EPOCH = 31

transform = transforms.ToTensor()


def load_validation_pairs(lr_dir, hr_dir, max_images):
    """
    Pre-load a random subset of LR/HR pairs as tensors (to CPU).
    Returns a list of (lr_tensor[1xCxHxW], hr_tensor[CxHxW]).
    """
    pairs = []
    lr_files = [f for f in os.listdir(lr_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not lr_files:
        print(f"Error: No LR images in {lr_dir}")
        return pairs

    subset = (np.random.choice(lr_files, max_images, replace=False)
              if len(lr_files) > max_images else lr_files)

    for fname in subset:
        hr_fname = fname.replace("x4", "")
        lr_path, hr_path = os.path.join(lr_dir, fname), os.path.join(hr_dir, hr_fname)
        if not os.path.exists(hr_path):
            continue
        lr_img = cv.imread(lr_path)
        hr_img = cv.imread(hr_path)
        if lr_img is None or hr_img is None:
            continue
        lr_tensor = transform(lr_img).unsqueeze(0)     # 1xCxHxW
        hr_tensor = transform(hr_img)                  # CxHxW
        pairs.append((lr_tensor, hr_tensor))
    print(f"✅ Pre-loaded {len(pairs)} validation pairs.")
    return pairs


def validate(model, val_pairs, cuda=False):
    """Validate on preloaded LR/HR tensor pairs."""
    psnr_scores, ssim_scores = [], []

    for lr_tensor, hr_tensor in val_pairs:
        if cuda:
            lr_tensor = lr_tensor.cuda()
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr_img_np = tensor_to_numpy_img(sr_tensor.squeeze(0))
        hr_img_np = tensor_to_numpy_img(hr_tensor)

        if min(sr_img_np.shape[:2]) < 7:
            continue
        sr_h, sr_w, _ = sr_img_np.shape
        hr_img_np_cropped = hr_img_np[:sr_h, :sr_w, :]

        psnr_scores.append(calc_psnr(hr_img_np_cropped, sr_img_np))
        ssim_scores.append(calc_ssim(hr_img_np_cropped, sr_img_np, win_size=3))

    if not psnr_scores:
        return None, None
    return np.mean(psnr_scores), np.mean(ssim_scores)


def monitor_loop():
    sr_model = EDSR(scale=scale)
    if cuda:
        sr_model = sr_model.cuda()

    # 🔹 Pre-load validation data once
    val_pairs = load_validation_pairs(val_dir, hr_val_dir, NUM_VALIDATION_IMAGES)
    if not val_pairs:
        print("No validation pairs found. Exiting.")
        return

    seen = set()
    psnr_history, ssim_history, epochs = [], [], []

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Performance Over Epochs")

    print("🔍 Monitoring checkpoints in", checkpoint_dir)
    while True:
        try:
            ckpts = [f for f in os.listdir(checkpoint_dir)
                     if f.endswith(".pt") or f.endswith(".pth")]
            ckpts.sort(key=lambda f: int(f.split("_")[-1].split(".")[0])
                       if "_" in f else 0)

            for ckpt in ckpts:
                try:
                    epoch_num = int(ckpt.split("_")[-1].split(".")[0])
                except ValueError:
                    continue
                if ckpt in seen or epoch_num < START_EPOCH:
                    continue

                path = os.path.join(checkpoint_dir, ckpt)
                print(f"\n📂 Found new checkpoint: {ckpt}")

                try:
                    checkpoint = torch.load(path,
                                            map_location="cpu",
                                            weights_only=True)
                    state_dict = checkpoint['model_state_dict']
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    sr_model.load_state_dict(state_dict)
                    sr_model.eval()
                except Exception as e:
                    print(f"Error loading {ckpt}: {e}")
                    continue

                psnr, ssim = validate(sr_model, val_pairs, cuda=cuda)
                if psnr is None:
                    continue

                print(f"✅ Validation — PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                epochs.append(epoch_num)
                psnr_history.append(psnr)
                ssim_history.append(ssim)

                if len(epochs) % 10 == 0:
                    ax[0].cla(); ax[1].cla()
                    ax[0].plot(epochs, psnr_history, 'o-', label="PSNR")
                    ax[1].plot(epochs, ssim_history, 'o-', label="SSIM", color="orange")
                    ax[0].set_title("Validation PSNR (dB)")
                    ax[1].set_title("Validation SSIM")
                    ax[0].set_xlabel("Epoch"); ax[1].set_xlabel("Epoch")
                    ax[0].grid(True); ax[1].grid(True)
                    ax[0].legend(); ax[1].legend()
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.1)

                seen.add(ckpt)

        except Exception as e:
            print(f"Monitoring loop error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    monitor_loop()
