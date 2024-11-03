import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def multiply(array):
    res = 1
    for el in array:
        res *= el
    return res


class Scheduler(torch.nn.Module):
    def __init__(self, T=1000, beta_min=1e-4, beta_max=0.02, device="cuda"):
        super().__init__()
        self.T = T
        self.device = device
        self.betas = np.linspace(beta_min, beta_max, T)

        self.alphas = 1 - self.betas

        self.alphas_bar = np.array(
            [multiply(self.alphas[: i + 1]) for i in range(T)], dtype=np.float32
        )

        self.betas_bar = torch.from_numpy(1 - self.alphas_bar).view(-1, 1).to(device)

    def noise(self, x, betas):
        noise = torch.randn_like(x)
        betas = betas.view(-1, 1, 1, 1).expand(x.shape)
        return torch.sqrt(1 - betas) * x + betas * noise, noise

    def get_blurred(self, x, return_steps=False):
        steps = torch.randint(0, self.T, size=(x.shape[0], 1), device=self.device)
        betas_bar = torch.gather(self.betas_bar, 0, steps)
        noised, noise = self.noise(x, betas_bar)
        if return_steps:
            return noised, noise, steps
        else:
            return noised, noise


if __name__ == "__main__":
    from imgen import Dataset

    device = "cuda"
    ds = Dataset("datasets/afhq")
    gen = ds.create_gen(batch_size=32, device=device)
    scheduler = Scheduler()
    imgs = next(gen)
    print(imgs.shape)
    noised, noise, Ts = scheduler.get_blurred(imgs, return_steps=True)
    print(Ts)
    noised = (noised + 1) / 2
    plt.imshow(noised[0].cpu().transpose(0, 2).transpose(0, 1))
    plt.show()
