import matplotlib.pyplot as plt
import torch
from utils.scheduler import Scheduler
from utils.inference import Inference
from models.unet import Unet
from utils.imgen import Dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import tqdm
device = "cuda:1"
model = Unet(device = device, channels=320).to(device)
print("num params : ",sum(p.numel() for p in model.parameters()))
model.load_state_dict(torch.load("model.pt"))
scheduler = Scheduler(device=device)
'''
ds_path = "datasets/afhq/train"
img_size = 128
ds = Dataset(ds_path,size = img_size)
batch = next(ds.create_gen(device=device))
'''
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1),
        transforms.Resize((128, 128)),
    ]
)



train_dataset = datasets.CelebA(root="data", transform=transform, download=True, split = "valid")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
def gene():
    for elt in tqdm.tqdm(train_loader):
        yield elt[0]


batch = next(gene()).to(device)
img_size = 128

inf = Inference(model,device= device,img_size =img_size)
print(batch.shape)

noised, noise, Ts = scheduler.get_blurred(batch, return_steps=True, t = 500)

noised = noised[0:1]
#show initial image , and initial blurred
plt.imshow(batch[0].permute(1,2,0).cpu().numpy())
plt.show()

plt.imshow(noised[0].permute(1,2,0).cpu().numpy())
plt.show()
t = scheduler.T - Ts[0].item() - 1 
print(t)

img = (inf(x_start = noised, t = t,show = True) + 1)/2

plt.imshow(img[0].permute(1,2,0).cpu().numpy())
plt.savefig("image_test2.png")