import torch
from torchvision.transforms import v2
import torchvision
import wandb
from models.unet import Unet
from utils.imgen import Dataset
from utils.scheduler import Scheduler
from utils.inference import Inference
import tqdm

def train_gene(ds):
        for elt in tqdm.tqdm(ds):
            yield elt[0]

def train_one_epoch(model, opt, ds, batch_size, scheduler, criterion, val_ds = None):
    #gen = ds.create_gen(batch_size=batch_size, loop_when_finish=False,device = device)
    train_gen = train_gene(ds)
    val_gen = train_gene(val_ds)
    train_loss = []
    val_loss = []
    for batch in train_gen:
        batch = batch.to(device)
        #with torch.autocast(device_type=device, dtype=torch.float16):
            #print(batch.shape)
        opt.zero_grad()
        imgs = batch
        noised, noise, Ts = scheduler.get_blurred(imgs, return_steps=True)
        pred = model(noised, Ts)

        l = criterion(pred, noise)
        l.backward()
        #scaler.scale(l).backward()
        opt.step()
        #scaler.step(opt)
        #scaler.update()
        train_loss.append(l.item())
        #wandb.log({"loss_step": l.item()})
    if val_ds != None:
        with torch.no_grad():    
            for batch in val_gen:
                batch = batch.to(device)
                imgs = batch
                noised, noise, Ts = scheduler.get_blurred(imgs, return_steps=True)
                pred = model(noised, Ts)

                l = criterion(pred, noise)

                val_loss.append(l.item())

    train_loss = sum(train_loss) / len(train_loss)  
    val_loss = sum(val_loss) / len(val_loss)
    wandb.log({"train_loss": train_loss,
               "val_loss": val_loss,
               })
    return train_loss,val_loss


if __name__ == "__main__":
    # write argparser

    # for the moment, we write all the parameters here
    ds_path = "datasets/afhq/train"
    device = "cuda:0"
    img_size = 128
    use_wandb = True
    batch_size = 32
    learning_rate = 8e-5
    num_epochs = 1000
    model = Unet(device=device,img_size=img_size,channels=320).to(device)
    print("num params : ",sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    '''
    ds = Dataset(ds_path,size = img_size)
    '''
    from utils.imgen2 import create_gen
    ds,val_ds = create_gen()
    scheduler = Scheduler(device=device)
    inf = Inference(model,device=device,img_size = img_size)
    criterion = torch.nn.MSELoss()
    #scaler = torch.amp.GradScaler()

    if use_wandb:
        wandb.init(project="Diffusion")
    loss = 0
    for epoch in range(num_epochs):
        #img = (inf().cpu() + 1) / 2
        #img = torchvision.utils.make_grid(img,nrow=1)

        train_loss,val_loss = train_one_epoch(model, opt, ds, batch_size, scheduler, criterion, val_ds)
        print(f"Epoch : {epoch+1}/{num_epochs}, train_loss : {loss}")

        torch.save(model.state_dict(), "model.pt")
