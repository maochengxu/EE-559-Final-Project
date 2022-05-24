from turtle import forward
from torch import empty
import torch
from torch.nn.functional import fold, unfold
from torch.nn import functional as F
from tqdm import tqdm
import pickle
import numpy as np
import math

   
from torch import empty
from torch.nn.functional import fold, unfold


class Module(object):
    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(stride) is int:
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if type(padding) is int:
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.bias = bias
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.weight = empty((out_channels, in_channels, *self.kernel_size)).normal_()/math.sqrt(in_channels+out_channels)
        self.bias = empty(out_channels).normal_()/math.sqrt(in_channels+out_channels)

        self.weight_grad = empty((out_channels, in_channels, *self.kernel_size))
        self.bias_grad = empty(out_channels)

        self.input_size = None
        self.input_features = None

    def forward(self, input):
        """
        :param input: (batch_size, in_channel, w, h)
        :return:
        """
        batch_size = input.shape[0]
        w, h = input.shape[2], input.shape[3]
        self.input_size = input.shape
        self.input_features = unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        wxb = self.weight.view(self.out_channels, -1) @ self.input_features + self.bias.view(1, -1, 1)
        return wxb.view(batch_size,
                        self.out_channels,
                        (w + 2 * self.padding[1] - self.kernel_size[0]) // self.stride[0] + 1,
                        (h + 2 * self.padding[0] - self.kernel_size[1]) // self.stride[1] + 1)

    def backward(self, gradwrtoutput):
        """
        :param gradwrtoutput: (batch_size, out_channels, w, h)
        :return:
        """
        batch_size = gradwrtoutput.shape[0]
        grad = gradwrtoutput.permute(0, 2, 3, 1).view((batch_size, -1, self.out_channels))
        delta_in = gradwrtoutput.permute(0, 2, 3, 1).reshape((-1, self.out_channels))
        self.weight_grad.add_(((self.input_features @ grad).sum(dim=0) / batch_size).permute(1, 0).view(self.weight.shape))
        self.bias_grad.add_(delta_in.sum(dim=0, keepdims=False) / batch_size)
        delta_out = fold((grad @ self.weight.view(self.out_channels, -1)).permute(0, 2, 1),
                         output_size=(self.input_size[2], self.input_size[3]),
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         stride=self.stride)
        return delta_out

    def param(self):
        return [[self.weight, self.weight_grad], [self.bias, self.bias_grad]]


class Upsampling(Module):
    def __init__(self, scale_factor=None) -> None:
        if type(scale_factor) is int:
            self.scale_factor = (scale_factor, scale_factor)
        else:
            self.scale_factor = scale_factor

    def forward(self, input):
        # upsampled = empty(input.shape[0], input.shape[1],
        #                   int(input.shape[2] * self.scale_factor[0]),
        #                   int(input.shape[3] * self.scale_factor[1]))
        # for i in range(input.shape[2]):
        #     for j in range(input.shape[3]):
        #         upsampled[:, :, i * self.scale_factor[0]: (i + 1) * self.scale_factor[0],
        #         j * self.scale_factor[1]: (j + 1) * self.scale_factor[1]] = input[:, :, i: i + 1, j: j + 1]
        # return upsampled
        return input.repeat_interleave(self.scale_factor[1], 3).repeat_interleave(self.scale_factor[0], 2)

    def backward(self, gradwrtoutput):
        # w, h = gradwrtoutput.shape[2] // self.scale_factor[0], gradwrtoutput.shape[3] // self.scale_factor[1]
        # downsampled = empty(gradwrtoutput.shape[0], gradwrtoutput.shape[1], w, h)
        # for p in range(gradwrtoutput.shape[0]):
        #     for q in range(gradwrtoutput.shape[1]):
        #         for i in range(w):
        #             for j in range(h):
        #                 downsampled[p, q, i, j] = gradwrtoutput[p, q,
        #                                                         i * self.scale_factor[0]: (i + 1) * self.scale_factor[0],
        #                                                         j * self.scale_factor[1]: (j + 1) * self.scale_factor[1]].sum()
        #
        # return downsampled
        return gradwrtoutput.unfold(3, self.scale_factor[1], self.scale_factor[1])\
            .sum(dim=4).unfold(2, self.scale_factor[0], self.scale_factor[0]).sum(dim=4)

    def param(self):
        return []

class NearestUpsampling(Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True) -> None:
        self.upsample = Upsampling(scale_factor)
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, input):
        return self.conv.forward(self.upsample.forward(input))

    def backward(self, gradwrtoutput):
        return self.upsample.backward(self.conv.backward(gradwrtoutput))

    def param(self):
        return self.conv.param()

class ReLU(Module):
    def __init__(self) -> None:
        self.input_features = None

    def forward(self, input):
        self.input_features = input.clone()
        return input.relu()

    def backward(self, gradwrtoutput):
        return (self.input_features >= 0) * gradwrtoutput

    def param(self):
        return []


class Sigmoid(Module):
    def __init__(self) -> None:
        self.input_sig = None

    def forward(self, input):
        self.input_sig = input.sigmoid()
        return self.input_sig

    def backward(self, gradwrtoutput):
        return self.input_sig * (1 - self.input_sig) * gradwrtoutput

    def param(self):
        return []


class MSE(Module):
    def __init__(self):
        self.input = None
        self.target = None

    def forward(self, input, target):
        self.input = input.clone()
        self.target = target.clone()
        return ((input - target) * (input - target)).mean()

    def backward(self):
        return 2 * (self.input - self.target) / self.input.view(-1).shape[0]

    def param(self):
        return []


class Sequential(Module):
    def __init__(self, *args: Module):
        self.modules = args

    def forward(self, input):
        output = input
        for module in self.modules:
            output = module.forward(output)
        return output

    def backward(self, gradwrtoutput):
        in_grad = gradwrtoutput
        for module in reversed(self.modules):
            in_grad = module.backward(in_grad)
        return in_grad

    def param(self):
        params = []
        for module in self.modules:
            if len(module.param()) > 0:
                params += module.param()
        return params


class SGD:
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, maximize=False):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.t = 0
        self.b = None

    def step(self):
        
        for param in self.params:
            if self.weight_decay != 0:
                param[1] += self.weight_decay * param[0]
            if self.momentum != 0:
                if self.t > 1:
                    self.b = self.momentum * self.b + (1 - self.dampening) * param[1]
                else:
                    self.b = param[1]
                if self.nesterov:
                    param[1] += self.momentum * self.b
                else:
                    param[1] = self.b
          #  print("before", param)
            if self.maximize:
                param[0] += self.lr * param[1]
            else:
                param[0] -= self.lr * param[1]
           # print("after", param)

    def zero_grad(self):
        for param in self.params:
            param[1].zero_()


### For mini-project 2
class Model():
    def __init__(self) -> None:
## instantiate model + optimizer + loss function + any other stuff you need
        conv1 = Conv2d(3, 8, 2, stride=2)
        conv2 = Conv2d(8, 8, 2, stride=2)
        # conv3 = Conv2d(8, 8, 4)
        # conv4 = Conv2d(8, 3, 8)
        upsampling1=NearestUpsampling(2,8,8,3,1,1)
        upsampling2=NearestUpsampling(2,8,3,3,1,1)

        # conv1.weight = torch.randn(3, 8, 2, 2)
        # conv2.weight = torch.randn(8, 8, 2, 2)
        # conv3.weight = torch.randn(8, 8, 4, 4)
        # conv4.weight = torch.randn(8, 3, 8, 8)

        self.layer=Sequential(conv1, ReLU(),
                         conv2, ReLU(),
                         upsampling1,  ReLU(),
                         upsampling2,Sigmoid())
        self.opt=SGD(self.layer.param(), 1.5)#lr=0.0001
        self.loss=MSE()
        self.batch_size=4

    def load_pretrained_model(self) -> None:
## This loads the parameters saved in bestmodel.pth into the model pass
        
        params = pickle.load(open( "bestmodel.pth", "rb" ))
        for i, param in enumerate(params):
            if len(param) > 0:
                if isinstance(self.layer.modules[i], Conv2d):
                    self.layer.modules[i].weight = param[0][0]
                    self.layer.modules[i].bias = param[1][0]
                elif isinstance(self.layer.modules[i], NearestUpsampling):
                    self.layer.modules[i].conv.weight = param[0][0]
                    self.layer.modules[i].conv.bias = param[1][0]                    
            
        # print(params_)

    def forward(self,x):
        x = self.layer.forward(x)
        return x

    def train(self, train_input, train_target, num_epochs) -> None:
        """_summary_

        Args:
            train_input (_type_): _description_
            train_target (_type_): _description_
            num_epochs (_type_): _description_
        """

        train_input = train_input.float() / 255.
        train_target = train_target.float() / 255.

        for e in range(num_epochs):
            acc_loss = 0
            for b in tqdm(range(0, train_input.size(0), self.batch_size)):
                self.opt.zero_grad()
                output = self.forward(train_input.narrow(0, b, self.batch_size))
                # print(self.layer.param())
                Loss = self.loss.forward(output, train_target.narrow(0, b, self.batch_size))
                
                acc_loss += Loss
                # self.layer.zero_grad() # zero all parameter gradients
                loss_grad = self.loss.backward()
                self.layer.backward(loss_grad)
                
                self.opt.step()

                if b % 10000 == 0:
                    print(e, acc_loss / 10000)
                    acc_loss = 0.
        # torch.save(self.layer.param.state_dict(), 'bestmodel.pth')
        params = []
        for module in self.layer.modules:
            params.append(module.param())
        pickle.dump(params, open("bestmodel.pth", "wb"))

    def predict(self, test_input) -> torch.Tensor:
        test_input = test_input.float() / 255.

        x = self.forward(test_input)

        return x*255.
#:test ̇input: tensor of size (N1, C, H, W) with values in range 0-255 that has to 
# be denoised by the trained or the loaded network.
#:returns a tensor of the size (N1, C, H, W) with values in range 0-255. pass

class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 8, 2, stride=2)
        self.conv3 = torch.nn.Conv2d(8, 8, 3,padding=1)
        self.conv4 = torch.nn.Conv2d(8, 3, 3, padding=1)
        self.upsample1 = torch.nn.Upsample(scale_factor=2)
        self.upsample2 = torch.nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.upsample1(y)
        y = F.relu(self.conv3(y))
        y = self.upsample2(y)
        y = F.sigmoid(self.conv4(y))

        return y


class ModelTorch():
    def __init__(self, lr=0.5,batch_size=50) -> None:
    ## instantiate model + optimizer + loss function + any other stuff you need
        # conv1 = torch.nn.Conv2d(3, 3, 2, stride=2)
        # conv2 = torch.nn.Conv2d(3, 3, 2, stride=2)
        # conv3 = torch.nn.Conv2d(3, 3, 4)
        # conv4 = torch.nn.Conv2d(3, 3, 8)
        
        # conv1.weight = torch.randn(3, 3, 2, 2)
        # conv2.weight = torch.randn(3, 3, 2, 2)
        # conv3.weight = torch.randn(3, 3, 4, 4)
        # conv4.weight = torch.randn(3, 3, 8, 8)

        # self.layer=torch.nn.Sequential(conv1, torch.nn.ReLU(),
        #                  conv2, torch.nn.ReLU(),
        #                  torch.nn.Upsample(2), conv3, torch.nn.ReLU(),
        #                  torch.nn.Upsample(3), conv4, torch.nn.Sigmoid())
    
        self.layer = SimpleModel()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer.to(self.device)
        # self.opt=SGD(self.layer.parameters(), 0.001)#lr=0.0001
        self.opt = torch.optim.SGD(self.layer.parameters(), lr=lr)
        # self.opt = torch.optim.Adam(self.layer.parameters(), 0.0001, (0.9, 0.99), 1e-7)
        self.loss=torch.nn.MSELoss()
        self.batch_size=batch_size

    def load_pretrained_model(self) -> None:
## This loads the parameters saved in bestmodel.pth into the model pass
        
        self.layer = torch.load('bestmodel.pth')

    # def forward(self,x):
    #     x = self.layer(x)
    #     return x

    def train(self, train_input, train_target, num_epochs) -> None:
#:train ̇input: tensor of size (N, C, H, W) containing a noisy version of the images 
# same images, which only differs from the input by their noise.
#:train ̇target: tensor of size (N, C, H, W) containing another noisy version of the
        # criterion = self.loss
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # optimizer = self.opt

        train_input = train_input.float() / 255.

        train_target = train_target.float() / 255.
        train_input = train_input.to(device)
        train_target = train_target.to(device)

        for e in range(num_epochs):
            acc_loss = 0.
            for b in tqdm(range(0, train_input.size(0), self.batch_size)):
                self.opt.zero_grad()
                x = train_input[b:b+self.batch_size]
                y = train_target[b:b+self.batch_size]
                output = self.layer(x)
                
                Loss = self.loss(output, y)
                acc_loss += Loss.item()

                
                Loss.backward()
                self.opt.step()
                if b % 10000 == 0:
                    print(e, acc_loss / 100)
                    acc_loss = 0.
        torch.save(self.layer, 'bestmodel.pth')
    def predict(self, test_input) -> torch.Tensor:
        test_input = test_input.float() / 255.

        x = self.layer(test_input)

        return x*255.
def compute_psnr(x, y, max_range=1.0):
    
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()
# def psnr(denoised , ground_truth):
# # Peak Signal to Noise Ratio: denoised and ground ̇truth have range [0, 1] 
#     mse = torch.mean((denoised - ground_truth) ** 2)
#     return -10 * torch.log10(mse + 10**-8)
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    noisy_imgs_1, noisy_imgs_2 = torch.load('/Applications/epfl/epfl2/DL/EE-559-Final-Project-main/Miniproject_2/Data/train_data.pkl')
    # n2n = Model(lr=i,batch_size=b)
    result=[]
    for i in np.arange(1.5,1.9,0.05):
        for b in range(4,8):
  
            n2n = Model()
            # n2n = ModelTorch(lr=i,batch_size=b)
            n2n.load_pretrained_model()
            n2n.train(noisy_imgs_1, noisy_imgs_2, 1)
            
            val_path = '/Applications/epfl/epfl2/DL/EE-559-Final-Project-main/Miniproject_2/Data/val_data.pkl'
            val_input, val_target = torch.load(val_path)
            val_target = val_target.float() / 255.0

            mini_batch_size = 4
            model_outputs = []
            for b in tqdm(range(0, val_input.size(0), mini_batch_size)):
                output = n2n.predict(val_input.narrow(0, b, mini_batch_size))
                model_outputs.append(output)
            model_outputs = torch.cat(model_outputs, dim=0)
            # print(model_outputs)
            print(model_outputs/255)
            print(val_target)
            output_psnr = compute_psnr(model_outputs/255., val_target)
            result.append(output_psnr)
            # output_psnr = compute_psnr(val_input/255.0, val_target)
            print(f"[PSNR {2}: {output_psnr:.2f} dB]")  
    print(result)