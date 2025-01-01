import torch
import torch.nn as nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim=64, heads=4, dim_head=16, dropout=0.):
        super().__init__()

        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj=nn.Linear(heads*dim_head,dim)
        

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out=self.proj(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim=64, heads=4, dim_head=16, dropout=0.):
        super().__init__()

        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(heads * dim_head, dim)


    def forward(self, x, memory):
        x = self.norm_q(x)
        
        memory = self.norm_kv(memory)

        q= self.to_q(x)
        
        k, v = self.to_kv(memory).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out=self.proj(out)
        return out
