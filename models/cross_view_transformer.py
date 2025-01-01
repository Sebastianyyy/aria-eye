
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from einops import rearrange
from .encoder import Encoder
from .decoder import Decoder
from torchvision import models

class CrossViewTransformer(nn.Module):
    def __init__(self, d_model=64, heads=4, enc_depth=4, dec_depth=4, grid_size=8, backbone_size=49, backbone_channels=512, dropout=0.1):
        super(CrossViewTransformer, self).__init__()
        pretrained_model = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.grid_size=grid_size
        hidden_dim = d_model

        self.encoder = Encoder(d_model=d_model, nhead=heads, num_layers=enc_depth,
                               dim_feedforward=hidden_dim * 4, dropout=dropout)
        self.decoder = Decoder(d_model=d_model, nhead=heads, num_layers=dec_depth,
                               dim_feedforward=hidden_dim * 4, dropout=dropout)
    
        self.proj_input = nn.Conv2d(backbone_channels, hidden_dim, 1)
        self.proj_output = nn.Linear(hidden_dim, 1)
        
        self.gaze_embed = nn.Parameter(torch.randn(1, grid_size**2, hidden_dim))
        self.pos_encoder = nn.Parameter(torch.randn(1, backbone_size, hidden_dim))
        
    def forward(self, x):
        #x: (B, C, H, W)
        if x.dim()==5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
        B, C, H, W = x.shape
        features=self.backbone(x)
        features=self.proj_input(features)
        
        features=rearrange(features, 'b c h w -> b (h w) c')
        features=features+self.pos_encoder
        
        enc=self.encoder(features)
        dec=self.decoder(self.gaze_embed,enc)
        out=self.proj_output(dec)
        # rearrange(x, 'b (n n) 1 -> b 1 n n', n=8)

        out=out.squeeze(-1).view(B, self.grid_size, self.grid_size).unsqueeze(1)
        
        return out
        
        
def get_model(config):
        return CrossViewTransformer(
            d_model=config['d_model'],
            heads=config['heads'], 
            enc_depth=config['enc_depth'], 
            dec_depth=config['dec_depth'], 
            grid_size=config['shape']
            )

        
        
        
        
