import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FeedForward, SelfAttention


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    SelfAttention(d_model, nhead, d_model // nhead, dropout=dropout),
                    FeedForward(d_model, dim_feedforward, dropout=dropout)
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
