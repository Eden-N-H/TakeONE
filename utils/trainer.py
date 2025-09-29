import torch.autograd
from torch import optim, nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Use relative imports, as this script is a module within a package
from .model_util import save_model, load_model
from .dataset import SRDataset

class Trainer:
    """
    The Trainer class encapsulates the entire training loop for the EDSR model.
    It handles data loading, model optimization, and checkpoint management.
    """
    def __init__(self, model, lr, batch_size, checkpoint_save_path, scale, lr_root, hr_root, cuda):
        """
        Initializes the Trainer with model, training parameters, and data paths.

        :param model: The EDSR model instance.
        :param lr: Learning rate for the optimizer.
        :param batch_size: Batch size for the DataLoader.
        :param checkpoint_save_path: Path to save model checkpoints.
        :param scale: The super-resolution scale factor.
        :param lr_root: The root directory for low-resolution images.
        :param hr_root: The root directory for high-resolution images.
        :param cuda: Boolean flag to enable GPU training.
        """
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.checkpoint_save_path = checkpoint_save_path
        self.cuda = cuda
        
        # Load the dataset
        self.dataset = SRDataset(lr_root=lr_root, hr_root=hr_root, scale=scale)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        # Define optimizer and loss function
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08
        )
        self.criterion = nn.L1Loss()
        
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()

    def train(self, start_epoch, total_epochs):
        """
        Runs the training loop for a specified number of epochs.

        :param start_epoch: The epoch to start training from (for resuming).
        :param total_epochs: The total number of epochs to train for.
        """
        print('Starting training...')
        for epoch in range(start_epoch, total_epochs):
            self.model.train()
            
            for i, (lr_img, hr_img) in enumerate(self.dataloader):
                # Move images to GPU if cuda is enabled
                if self.cuda:
                    lr_img = lr_img.cuda()
                    hr_img = hr_img.cuda()
                
                # Perform forward pass
                sr_img = self.model(lr_img)
                
                
                # Calculate loss
                loss = self.criterion(sr_img, hr_img)
                
                # Zero out gradients, perform backpropagation, and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch [{epoch+1}/{total_epochs}], Batch [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

            # Save model checkpoint after each epoch
            save_model(self.model, self.optimizer, epoch + 1, self.checkpoint_save_path)
