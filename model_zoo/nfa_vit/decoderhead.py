import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class Multiple(nn.Module):
    def __init__(self, 
                 init_value = 1e-6,
                 embed_dim = 512,
                 predict_channels = 1,
                 norm_layer = partial(nn.LayerNorm, eps=1e-6) ):
        super(Multiple, self).__init__()
        self.gamma1 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma2 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma3 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma4 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma5 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma6 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        # self.drop_path = nn.Identity()
        self.norm = norm_layer(embed_dim)
        
        self.conv_layer1 = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer3 = nn.Conv2d(in_channels=160, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(embed_dim, predict_channels, kernel_size= 1)
    def forward(self, x):
        c1, c2, c3, c4 = x
        
        c1 = self.conv_layer1(c1)
        c2 = self.conv_layer2(c2)
        c3 = self.conv_layer3(c3)
        c4 = self.conv_layer4(c4)
        
        b, c, h, w = c1.shape
        c2 = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False)
        
        
        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2) 
        x = self.gamma1*c1 + self.gamma2*c2 + self.gamma3*c3 + self.gamma4*c4
        x= x.transpose(1, 2).reshape(b, c, h, w)
        x = (self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).contiguous()
        x = self.conv_last(x)
        return x