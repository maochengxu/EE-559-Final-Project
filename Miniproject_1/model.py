import torch
from torch import nn
from torch import functional as F
from torch import optim

from .others.unetwork import UNetwork

class Model():
    def __init__(self) -> None:
        """instantiate model + optimizer + loss function
        """
        self.unet = UNetwork(in_channels=3)
        self.opt = optim.Adam(self.unet.parameters(), 0.001, (0.9, 0.99), 1e-8)
        self.loss = nn.MSELoss()

    def load_pretrained_model(self) -> None:
        """Loads the parameters saved in bestmodel.pth into the model
        """
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        """Train the model

        Args:
            train_input (torch.tensor): size (N, C, H, W) containing a noisy version of the image
            train_target (torch.Tensor): size (N, C, H, W) containing another noisy version of the same images
            num_epochs (int): number of epochs
        """
        pass

    def predict(self, test_input) -> torch.Tensor:
        """Use the model to predict

        Args:
            test_input (torch.Tensor): size (N1, C, H, W) that has to be denoised by the trained or the loaded network

        Returns:
            torch.Tensor: size (N1, C, H, W)
        """
        pass
