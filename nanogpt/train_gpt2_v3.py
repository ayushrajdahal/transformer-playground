import torch, tiktoken, math, time
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size:int = 50304
    block_size:int = 16
    n_embd:int = 768
    n_layer:int = 12
    n_head:int = 12

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.proj = nn.Linear(config.n_embd, 3*config.n_embd)
        
    def forward(self, x):
        pass

# class DataLoader:


class Block(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=12)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config=config
    def forward(self, idx):
        pass