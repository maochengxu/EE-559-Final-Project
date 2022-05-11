import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
# from torchsummary import summary

from others.unetwork import UNetwork

# TODO Tensorboard

class Model():
    def __init__(self) -> None:
        """instantiate model + optimizer + loss function + GPU support
        """
        self.unet = UNetwork(in_channels=3)
        self.opt = optim.Adam(self.unet.parameters(), 0.001, (0.9, 0.99), 1e-8)
        self.loss = nn.MSELoss()
        self.batch_size = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet.to(self.device)

    def load_pretrained_model(self) -> None:
        """Loads the parameters saved in bestmodel.pth into the model
        """
        self.unet = torch.load('./bestmodel.pth')
        self.unet.eval()

    def train(self, train_input, train_target, num_epochs) -> None:
        """Train the model

        Args:
            train_input (torch.tensor): size (N, C, H, W) containing a noisy version of the image
            train_target (torch.Tensor): size (N, C, H, W) containing another noisy version of the same images
            num_epochs (int): number of epochs
        """
        # TODO Add data augmentation features
        self.unet.train(True)
        # Use dataloader to create mini-batch
        tarin_pair = torch.stack((train_input / 255., train_target / 255.), dim=1) # Create (input, target) pair (N, 2, C, H, W)
        train_loader = DataLoader(tarin_pair, batch_size=self.batch_size, shuffle=True)

        print("Training Started!")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for idx, img_pairs in enumerate(train_loader):
                source, target = torch.chunk(img_pairs, 2, 1)
                source = source.squeeze(dim=1).to(self.device)
                target = target.squeeze(dim=1).to(self.device)

                source_output = self.unet(source)

                loss = self.loss(source_output, target)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                running_loss = running_loss + loss.item()
                if (idx + 1) % 100 == 0:
                    print('Epoch %d, Batch %d >>>>>>>>>>>> Loss: %.3f' % (epoch, idx + 1, running_loss / 100))
                    running_loss = 0.0
        print('Training Finished!')
        torch.save(self.unet, 'bestmodel.pth')


    def predict(self, test_input) -> torch.Tensor:
        """Use the model to predict

        Args:
            test_input (torch.Tensor): size (N1, C, H, W) that has to be denoised by the trained or the loaded network

        Returns:
            torch.Tensor: size (N1, C, H, W)
        """
        self.unet.eval()
        output = self.unet(test_input)
        return output


# if __name__ == "__main__":
#     noisy_imgs_1, noisy_imgs_2 = torch.load('./Data/train_data.pkl')
#     n2n = Model()
#     n2n.train(noisy_imgs_1, noisy_imgs_2, 10)
