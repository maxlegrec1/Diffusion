import math

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from models.unet import Unet
from utils.scheduler import Scheduler


class Inference(torch.nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.model = Unet().to(device)
        self.model.load_state_dict(torch.load("model.pt"))
        self.scheduler = Scheduler()
        self.device = device

    @torch.no_grad()
    def forward(self, num_images=1):

        x_t = torch.randn((num_images, 3, 512, 512), device=self.device)

        for i in tqdm(range(0, self.scheduler.T)):
            t = self.scheduler.T - 1 - i
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
            if t != 0:
                x_t = x_t + math.sqrt(self.scheduler.betas[t]) * torch.randn_like(x_t)

        return x_t
