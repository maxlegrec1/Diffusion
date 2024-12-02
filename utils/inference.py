import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from models.unet import Unet
from utils.scheduler import Scheduler
import torchvision

class Inference(torch.nn.Module):
    def __init__(self, model, device="cuda",img_size = 256):
        super().__init__()
        self.model = model
        self.scheduler = Scheduler()
        self.device = device
        self.img_size = img_size
    @torch.no_grad()
    def forward(self, num_images=1,show = False,x_start = None,t = 0):
        if t != 0:
            x_t = x_start
        else:
            x_t = torch.randn((num_images, 3, self.img_size, self.img_size), device=self.device)

        for i in tqdm(range(t, self.scheduler.T)):
            t = self.scheduler.T - 1 - i
            t = torch.from_numpy(np.array([t])).view(1,1).to(self.device)
            x_t = (
                1
                / math.sqrt(self.scheduler.alphas[t])
                * (
                    x_t
                    - (1 - self.scheduler.alphas[t])
                    * self.model(x_t, t)
                    / (math.sqrt(1 - self.scheduler.alphas_bar[t]))
                )
            )
            if show and i%100 == 0:
                grid = torchvision.utils.make_grid((x_t.cpu() + 1) / 2,nrow=1)
                grid = grid.permute(1,2,0)
                plt.imshow(grid)
                plt.show()
            if t != 0:
                x_t = x_t + math.sqrt(self.scheduler.betas[t]) * torch.randn_like(x_t)

        return x_t


if __name__ == "__main__":
    device = "cuda"
    from models.unet import Unet
    from utils.imgen import Dataset
    model = Unet(device = device).to(device)
    ds_path = "datasets/afhq/train"
    ds = Dataset(ds_path)
    scheduler = Scheduler(device=device)
    batch = next(ds.create_gen(device=device))
    print(batch.shape)