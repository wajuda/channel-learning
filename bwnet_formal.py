import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv0 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class Res2Block(nn.Module):
    def __init__(self, channel, ms_dim, num_resblocks):
        super().__init__()
        self.block = nn.ModuleList([])
        for _ in range(num_resblocks):
            self.block.append(ResBlock(channel))
        self.to_hrms = nn.Sequential(
            nn.Conv2d(channel, ms_dim, 3, 1, 1)
        )

    def forward(self, x):
        skip = x
        for resblock in self.block:
            x = resblock(x)
        x = x + skip
        output = self.to_hrms(x)
        return x, output


class Attention(nn.Module):
    def __init__(self, dim, num_res2blocks, dropout=0.):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.hidden_dim = dim * 2
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, num_res2blocks),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        attn = (q.transpose(-2, -1) @ k) * self.temperature // (n // (64 * 64))
        out = self.to_out(attn).softmax(dim=-1)
        return out.transpose(-2, -1)


class BandSelectBlock(nn.Module):
    def __init__(self, dim, num_res2blocks):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_res2blocks)
        self.temperature = nn.Parameter(torch.zeros(1, 8, 1, 1))

    def forward(self, input, x):
        H = input.shape[2]
        input = self.norm(rearrange(input, 'B C H W -> B (H W) C', H=H))
        attn = self.attn(input)  # B x num_res2blocks x C
        x = torch.stack(x, dim=0).transpose(0, 1)  # B x num_res2blocks x C x H x W
        output = torch.sum(attn.unsqueeze(-1).unsqueeze(-1) * x, dim=1) + self.temperature  # B x C x H x W
        return output


class BWNet(nn.Module):
    def __init__(self, pan_dim=1, ms_dim=8, channel=24, num_resblocks=2, num_res2blocks=4):
        super().__init__()

        self.ms_dim = ms_dim
        self.upsample = nn.Sequential(
            nn.Conv2d(ms_dim, ms_dim * 16, 3, 1, 1),
            nn.PixelShuffle(4)
        )
        self.raise_dim = nn.Sequential(
            nn.Conv2d(ms_dim, channel, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_res2blocks):
            self.layers.append(Res2Block(channel, ms_dim, num_resblocks))
        self.to_output = BandSelectBlock(ms_dim, num_res2blocks)

    def forward(self, ms, pan):
        ms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        pan = pan.repeat(1, self.ms_dim, 1, 1)
        input = pan - ms
        x = self.raise_dim(input)
        output_list = []
        for layer in self.layers:
            x, output = layer(x)
            output_list.append(output)
        output = self.to_output(input, output_list)
        return output + ms


def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

model = BWNet().cuda()
summaries(model, grad=True)







