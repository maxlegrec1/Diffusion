import math

import numpy as np
import torch


def create_positional_encoding(max_t=1000, t_dim=128):
    pe = torch.zeros(max_t, t_dim)
    position = torch.arange(0, max_t, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, t_dim, 2).float() * (-math.log(10000.0) / t_dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class DownBlock(torch.nn.Module):
    def __init__(self, in_size, num_channels, device = "cuda"):
        super().__init__()
        #print(f"Down : {in_size}, {num_channels}")
        num_channels = min(num_channels,640)
        self.num_channels = num_channels
        if in_size == 64:
            self.attn = True
            self.attn_size = int(in_size//2)
            self.pos_enc = create_positional_encoding(max_t = self.attn_size * self.attn_size,t_dim = min(640,num_channels*2 )).to(device).unsqueeze(0)
            self.theta = torch.nn.Conv2d(
            in_channels=min(num_channels*2,640),
            out_channels=min(num_channels*2,640),
            kernel_size=1,
            padding=0,
            )
            self.phi = torch.nn.Conv2d(
            in_channels=min(num_channels*2,640),
            out_channels=min(num_channels*2,640),
            kernel_size=1,
            padding=0,
            )
            self.g = torch.nn.Conv2d(
            in_channels=min(num_channels*2,640),
            out_channels=min(num_channels*2,640),
            kernel_size=1,
            padding=0,
            )
        else:
            self.attn = False
        self.num_channels = num_channels
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )


        self.conv4 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=min(num_channels*2,640),
            kernel_size=3,
            padding=1,
        )
        self.norm1 = torch.nn.GroupNorm(16,num_channels)
        self.norm2 = torch.nn.GroupNorm(16,min(640,num_channels * 2))
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        #print(    f"DownBlock :  In size : {x.shape}" )
        # First convolution block

        h = self.conv1(x)
        h= self.conv2(x) + x
        s = h
        h = self.norm1(h)
        h = self.activation(h)
        
        # Second convolution block
        h = self.conv3(h)+s
        h = self.conv4(h)
        h = self.norm2(h)
        h = self.activation(h)

        # Apply pooling for the next layer
        h = self.pool(h)
        # print(f"DpwnBlock : out_size : {h.shape}")

        if self.attn:
            h = h.view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size).permute(0,2,1)+ self.pos_enc.expand(h.shape[0],-1,-1)
            h = h.permute(0,2,1).view(-1,min(640,self.num_channels*2),self.attn_size,self.attn_size)#back to BCHW
            theta = self.theta(h).view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size).permute(0,2,1)#B,HW,C
            phi = self.phi(h).view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size)#B,C,HW
            #print(theta.shape,phi.shape)
            prod = torch.bmm(theta,phi) #B,HW,HW
            softmaxed = torch.softmax(prod,dim = -1)

            g = self.g(h).view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size).permute(0,2,1)
            h = torch.bmm(softmaxed,g).view(-1,self.attn_size,self.attn_size,min(640,self.num_channels*2))
            h = h.permute(0,3,1,2)#B,C,H,W
        return h, h


class UpBlock(torch.nn.Module):
    def __init__(self, out_size, num_channels, device = "cuda"):
        super().__init__()
        num_channels = min(num_channels,640)
        self.num_channels = num_channels
        #print(f"Up : {out_size}, {num_channels}")
        if out_size == 64:
            self.attn = True
            self.attn_size = int(out_size//2)
            self.pos_enc = create_positional_encoding(max_t = self.attn_size * self.attn_size,t_dim = min(640,num_channels*2 ) ).to(device).unsqueeze(0)
            self.theta = torch.nn.Conv2d(
            in_channels= min(num_channels*2,640),
            out_channels=min(num_channels*2,640),
            kernel_size=1,
            padding=0,
            )
            self.phi = torch.nn.Conv2d(
            in_channels=min(num_channels*2,640),
            out_channels=min(num_channels*2,640),
            kernel_size=1,
            padding=0,
            )
            self.g = torch.nn.Conv2d(
            in_channels=min(num_channels*2,640),
            out_channels=min(num_channels*2,640),
            kernel_size=1,
            padding=0,
            )
        else:
            self.attn = False
        self.num_channels = num_channels
        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=min(num_channels*2,640),
            out_channels=num_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = torch.nn.GroupNorm(16,num_channels)
        self.norm2 = torch.nn.GroupNorm(16,num_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        #print(f"UpBlock :  In size : {x.shape}, {self.num_channels}")
        if self.attn:
            h = x
            h = h.view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size).permute(0,2,1)+ self.pos_enc.expand(h.shape[0],-1,-1)
            h = h.permute(0,2,1).view(-1,min(640,self.num_channels*2),self.attn_size,self.attn_size)#back to BCHW
            theta = self.theta(h).view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size).permute(0,2,1)#B,HW,C
            phi = self.phi(h).view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size)#B,C,HW
            #print(theta.shape,phi.shape)
            prod = torch.bmm(theta,phi) #B,HW,HW
            softmaxed = torch.softmax(prod,dim = -1)

            g = self.g(h).view(-1,min(640,self.num_channels*2),self.attn_size*self.attn_size).permute(0,2,1)
            h = torch.bmm(softmaxed,g).view(-1,self.attn_size,self.attn_size,min(640,self.num_channels*2))
            x = h.permute(0,3,1,2)#B,C,H,W
        # Upsample
        x = self.upsample(x)

        # First convolution block
        h = self.conv1(x)
        h = self.conv2(x) + x
        s = h 
        h = self.norm1(h)
        h = self.activation(h)

        # Second convolution block
        h = self.conv3(h)
        h = self.conv4(h)+ s
        h = self.norm2(h)
        h = self.activation(h)
        # print(f"UpBlock :  Out size : {h.shape}")
        return h


class Unet(torch.nn.Module):

    def __init__(self, img_size=128, channels=224, latent_space_dim=None,device = "cuda", f = 8):
        super().__init__()
        if latent_space_dim == None:
            latent_space_dim = img_size // f
        self.latent_space_dim = latent_space_dim
        self.pe = create_positional_encoding().to(device)
        self.num_blocks = int(np.log2(img_size // latent_space_dim))
        #print(f"num_blocks : {self.num_blocks}")
        # Add initial convolution to handle input channels
        self.initial_conv = torch.nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.linears_down = torch.nn.ModuleList(
            [
                torch.nn.Linear(128, min(640,int(channels * math.pow(2, i))))
                for i in range(self.num_blocks)
            ]
        )
        self.linear_first = torch.nn.Linear(128, 128)
        self.linears_up = torch.nn.ModuleList(
            [
                torch.nn.Linear(128, min(640,int(channels * math.pow(2, i) * 2)))
                for i in range(self.num_blocks)
            ]
        )
        self.down_blocks = torch.nn.ModuleList(
            [
                DownBlock(
                    in_size=img_size // math.pow(2, i),
                    num_channels=int(channels * math.pow(2, i)),
                    device = device,
                )
                for i in range(self.num_blocks)
            ]
        )
        self.up_blocks = torch.nn.ModuleList(
            [
                UpBlock(
                    out_size=img_size // math.pow(2, i),
                    num_channels=int(channels * math.pow(2, i)),
                    device = device,
                )
                for i in range(self.num_blocks)
            ]
        )

        # Add final convolution to map back to 3 channels
        self.final_conv = torch.nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x_t, Ts):
        # Initial convolution
        x_t = self.initial_conv(x_t)
        Ts_embed = self.pe[
            Ts.view(
                -1,
            )
        ]
        Ts_embed = torch.nn.functional.relu(self.linear_first(Ts_embed))
        res = []
        # encoding
        for i in range(self.num_blocks):
            x_t = x_t + torch.nn.functional.relu(self.linears_down[i](Ts_embed)).view(
                x_t.shape[0], x_t.shape[1], 1, 1
            )
            x_t, s_i = self.down_blocks[i](x_t)
            res.append(s_i)
        #print(x_t.shape)
        # don't put last skip
        res[-1] = torch.zeros_like(x_t)
        # decoding
        for i in range(self.num_blocks):
            x_t = x_t + torch.nn.functional.relu(
                self.linears_up[self.num_blocks - 1 - i](Ts_embed)
            ).view(x_t.shape[0], x_t.shape[1], 1, 1)
            x_t = self.up_blocks[self.num_blocks - 1 - i](
                x_t + res[self.num_blocks - 1 - i]
            )

        # Final convolution to get back to image channels
        x_t = self.final_conv(x_t)

        return x_t


if __name__ == "__main__":
    model = Unet()
    print(sum(p.numel() for p in model.parameters()))
    test_input = torch.zeros((1, 3, 256, 256))
    output = model(test_input)
    print(output.shape)
