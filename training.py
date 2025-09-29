import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.edsr import EDSR
from utils.trainer import Trainer
from utils.evaluation_util import calc_psnr, calc_ssim, tensor_to_numpy_img
from utils.model_util import inference


def main():
    """
    Train or fine-tune the EDSR model.  If --resume_checkpoint is given,
    --epochs means the *additional* epochs to run.
    """
    parser = argparse.ArgumentParser(description='EDSR Super-Resolution Training')
    # model configuration
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor')
    parser.add_argument('--num_residual_blocks', type=int, default=32, help='Number of residual blocks')
    parser.add_argument('--num_channels', type=int, default=256, help='Number of feature channels')

    # training configuration
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate (default lower for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run. If resuming, this is *additional* epochs.')

    # data & checkpoint paths
    parser.add_argument('--lr_dir', type=str, required=True, help='Path to the low-resolution images directory')
    parser.add_argument('--hr_dir', type=str, required=True, help='Path to the high-resolution images directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume training from')

    # hardware
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training')
    args = parser.parse_args()

    # device setup
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f"Using device: {device}")

    # model & trainer
    model = EDSR(scale=args.scale).to(device)
    trainer = Trainer(
        model=model,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_save_path=args.checkpoint_dir,
        scale=args.scale,
        lr_root=args.lr_dir,
        hr_root=args.hr_dir,
        cuda=args.cuda
    )

    # checkpoint resume logic
    start_epoch = 0
    total_epochs = args.epochs
    if args.resume_checkpoint:
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            total_epochs = start_epoch + args.epochs   # add extra epochs
            print(f"Resuming from epoch {start_epoch}, will train until epoch {total_epochs}...")
        except FileNotFoundError:
            print(f"Checkpoint not found at {args.resume_checkpoint}. Starting new training.")
    else:
        print("Starting a new training session...")

    # training
    trainer.train(start_epoch=start_epoch, total_epochs=total_epochs)

    # quick evaluation
    print("\nStarting evaluation...")
    lr_sample, hr_sample = trainer.dataset[0]
    lr_sample, hr_sample = lr_sample.to(device), hr_sample.to(device)
    sr_img_tensor = inference(model, lr_sample)
    hr_img_np = tensor_to_numpy_img(hr_sample)
    sr_img_np = tensor_to_numpy_img(sr_img_tensor)
    psnr_value = calc_psnr(hr_img_np, sr_img_np)
    ssim_value = calc_ssim(hr_img_np, sr_img_np)
    print(f"Evaluation Results:\nPSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.4f}")


if __name__ == '__main__':
    main()
