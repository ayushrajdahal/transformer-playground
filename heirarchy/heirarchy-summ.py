import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self, dim, rank=2, seq_length=4, conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1, bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1, bias=False))
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_h -= seq_length // 2
                relative_coords_h /= (seq_length // 2)
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
                self.relative_bias = self.pos_emb
            else:
                seq_length = int(seq_length**0.5)
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
                relative_coords_table -= seq_length // 2
                relative_coords_table /= (seq_length // 2)
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1, 2))
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
        input_tensor = input_tensor + self.pos_emb
        return input_tensor

class FasterViTText(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, seq_length, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PosEmbMLPSwinv1D(dim=embed_dim, rank=1, seq_length=seq_length)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) for _ in range(num_layers)
        ])
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert tokens to embeddings
        x = self.pos_embedding(x)  # Add positional embeddings

        # Transformer Encoder Layers
        for layer in self.layers:
            x = layer(x)

        # Attention Mechanism
        x = x.transpose(0, 1)  # Shape (seq_length, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.transpose(0, 1)  # Shape (batch_size, seq_length, embed_dim)

        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)  # Final classification layer
        return x

# Example usage
vocab_size = 10000
embed_dim = 256
num_heads = 8
seq_length = 128
num_layers = 6
num_classes = 10

model = FasterViTText(vocab_size, embed_dim, num_heads, seq_length, num_layers, num_classes)
input_tensor = torch.randint(0, vocab_size, (32, seq_length))  # Batch of 32 sequences, each of length 128
output = model(input_tensor)
print(output.shape)  # Should be (32, 10)
