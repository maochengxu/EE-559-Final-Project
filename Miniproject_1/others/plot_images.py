import torch
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n2n = Model()
n2n.load_pretrained_model()
val_path = "./Data/val_data.pkl"
val_input, val_target = torch.load(val_path)
val_input = (val_input / 255.).to(device)
val_target = (val_target / 255.).numpy()
for i in range(5):
    output = n2n.predict(val_input[i].reshape((1, 3, 32, 32))) / 255.
    img_output = (output.detach().cpu().numpy().squeeze()).T
    img_target = val_target[i].T
    plt.imsave('./Miniproject_1/others/imgs/output_' + str(i) + '.png', img_output)
    plt.imsave('./Miniproject_1/others/imgs/target_' + str(i) + '.png', img_target)
