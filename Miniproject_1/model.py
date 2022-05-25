import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from .others.unetwork import UNetwork
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time


class Model():
    def __init__(self) -> None:
        """instantiate model + optimizer + loss function + GPU support
        """
        self.unet = UNetwork(in_channels=3)
        self.opt = optim.Adam(self.unet.parameters(), 0.001, (0.9, 0.99), 1e-8)
        self.loss = nn.MSELoss()
        self.batch_size = 100
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.unet.to(self.device)
        self.tb = False
        self.save = True

    def load_pretrained_model(self) -> None:
        """Loads the parameters saved in bestmodel.pth into the model
        """
        model_path = Path(__file__).parent / "bestmodel.pth"
        unet_state_dict = torch.load(model_path, map_location=self.device)
        self.unet.load_state_dict(unet_state_dict)
        # self.unet = torch.load('./Miniproject_1/bestmodel.pth')
        # self.unet = torch.load('/home/paperspace/Project/EE-559-Final-Project/Miniproject_1/bestmodel.pth')
        self.unet.to(self.device)
        self.unet.eval()

    def train(self, train_input, train_target, num_epochs) -> None:
        """Train the model

        Args:
            train_input (torch.tensor): size (N, C, H, W) containing a noisy version of the image
            train_target (torch.Tensor): size (N, C, H, W) containing another noisy version of the same images
            num_epochs (int): number of epochs
        """
        self.unet.train(True)
        if train_input.max() > 1:
            train_input = train_input / 255.
        if train_target.max() > 1:
            train_target = train_target / 255.
        # Use dataloader to create mini-batch
        # Create (input, target) pair (N, 2, C, H, W)
        tarin_pair = torch.stack((train_input, train_target), dim=1)
        train_loader = DataLoader(
            tarin_pair, batch_size=self.batch_size, shuffle=True)

        print("Training Started!")
        if self.tb:
            writer = SummaryWriter(log_dir="./Miniproject_1/others/runs")
            writer.add_graph(self.unet, input_to_model=torch.randn(
                (1, 3, 32, 32)).to(self.device))
        idx_num = 0
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_psnr = 0.0
            for idx, img_pairs in enumerate(train_loader):
                source, target = torch.chunk(img_pairs, 2, 1)
                source = source.squeeze(dim=1).to(self.device)
                target = target.squeeze(dim=1).to(self.device)

                source_output = self.unet(source)

                loss = self.loss(source_output, target)
                psnr = self.compute_psnr(source_output, target)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                running_loss = running_loss + loss.item()
                running_psnr = running_psnr + psnr
                idx_num = idx_num + 1
                if idx % 100 == 99:
                    print('Epoch %d, Batch %d >>>>>>>>>>>> Loss: %.3f, PSNR: %.3f dB' % (
                        epoch, idx + 1, running_loss / 100, running_psnr / 100))
                    if self.tb:
                        writer.add_scalar(tag='Loss/train',
                                          scalar_value=running_loss / 100,
                                          global_step=idx_num)
                        writer.add_scalar(tag='PSNR/train',
                                          scalar_value=running_psnr / 100,
                                          global_step=idx_num)
                    running_loss = 0.0
                    running_psnr = 0.0
        if self.tb:
            writer.close()
        print('Training Finished!')
        if self.save:
            torch.save(self.unet.state_dict(), './Miniproject_1/bestmodel.pth')

    def predict(self, test_input) -> torch.Tensor:
        """Use the model to predict

        Args:
            test_input (torch.Tensor): size (N1, C, H, W) that has to be denoised by the trained or the loaded network

        Returns:
            torch.Tensor: size (N1, C, H, W)
        """
        self.unet.eval()
        if test_input.max() > 1:
            test_input = test_input / 255.
        test_input = test_input.to(self.device)
        output = self.unet(test_input) * 255.
        return output

    @staticmethod
    def compute_psnr(x, y, max_range=1.0):
        assert x.shape == y.shape and x.ndim == 4
        return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1, 2, 3))).mean()


if __name__ == "__main__":
    noisy_imgs_1, noisy_imgs_2 = torch.load('./Data/train_data.pkl')
    n2n = Model()
    t1 = time.time()
    n2n.train(noisy_imgs_1, noisy_imgs_2, 1)
    t2 = time.time()
    print('Training lasts %.1f s' % (t2 - t1))
