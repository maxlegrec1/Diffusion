from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1),
        transforms.Resize((128, 128)),
    ]
)



train_dataset = datasets.CelebA(root="data", transform=transform, download=True, split = "train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset  = datasets.CelebA(root="data", transform=transform, download=True, split = "valid")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

def create_gen():
    return train_loader,val_loader