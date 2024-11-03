import torch
from torchvision.transforms import v2

import wandb
from models.unet import Unet
from utils.imgen import Dataset
from utils.scheduler import Scheduler


def train_one_epoch(model, opt, ds, batch_size, scheduler, criterion):
    gen = ds.create_gen(batch_size=batch_size, loop_when_finish=False)
    loss_epoch = []
    for batch in gen:
        opt.zero_grad()
        imgs = batch
        noised, noise, Ts = scheduler.get_blurred(imgs, return_steps=True)
        pred = model(noised, Ts)

        l = criterion(pred, noise)

        l.backward()
        opt.step()
        loss_epoch.append(l.item())
        wandb.log({"loss_step": l.item()})
    return sum(loss_epoch) / len(loss_epoch)


if __name__ == "__main__":
    # write argparser

    # for the moment, we write all the parameters here
    ds_path = "datasets/afhq"
    device = "cuda"
    use_wandb = True
    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 1000
    model = Unet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ds = Dataset(ds_path)
    scheduler = Scheduler()
    criterion = torch.nn.MSELoss()
    if use_wandb:
        wandb.init(project="Diffusion")

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, opt, ds, batch_size, scheduler, criterion)
        print(f"Epoch : {epoch+1}/{num_epochs}, Loss : {loss}")
        if use_wandb:
            wandb.log({"loss": loss})
        torch.save(model.state_dict(), "model.pt")
