import torch
from tqdm import tqdm
import sys
sys.path.append("/home/paperspace/Project/EE-559-Final-Project/Miniproject_1")
from model import Model
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()


def test_model_pnsr(model, project_number):
    model.load_pretrained_model()
    model.unet.to(device)

    val_path = "./Data/val_data.pkl"
    val_input, val_target = torch.load(val_path)

    val_input = val_input.float().to(device) / 255.0
    val_target = val_target.float().to(device) / 255.0
    with torch.no_grad():
        output_psnr_before = compute_psnr(val_input, val_target)
        mini_batch_size = 100
        model_outputs = []
        for b in tqdm(range(0, val_input.size(0), mini_batch_size)):
            output = model.predict(val_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output.cuda())
        model_outputs = torch.cat(model_outputs, dim=0) / 255.
        output_psnr = compute_psnr(model_outputs, val_target)
    print(f"[PSNR BEFORE: {output_psnr_before:.2f} dB]")
    print(f"[PSNR AFTER: {output_psnr:.2f} dB]")


n2n = Model()
test_model_pnsr(n2n, 1)
out = n2n.predict(torch.rand(1, 3, 512, 512) * 255)
print(out.max())
# n2n.load_pretrained_model()
# val_path = "./Data/val_data.pkl"
# val_input, val_target = torch.load(val_path)
# val_input = (val_input/255.).to(device)
# output = n2n.predict(val_input[0].reshape((1, 3, 32, 32)))
# plt.imshow((output.detach().cpu().squeeze()).T)
