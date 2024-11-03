import matplotlib.pyplot as plt
import torch

from utils.inference import Inference

inf = Inference("model.pt")
img = (inf()[0].transpose(0, 2).transpose(0, 1).cpu() + 1) / 2

plt.imshow(img)
plt.show()
