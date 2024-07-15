import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# text = load_dataset("afmck/text8-chunked1024")

# read file
with open('../input.txt', 'r') as f:
    text = f.read()

# cuda stuff
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
print("using device:", device)

# ensure reproducibility
torch.manual_seed(1337)

# encode/decode
stoi = {ch: i for i, ch in enumerate(sorted(list(set(text))))}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# hyperparameters
vocab_size = len(stoi)
batch_size = 16
n_head = 8
n_embd = 40
# ~added~
block_size = 32
max_iters = 5000
eval_interval = 100
eval_iters = 200
learning_rate = 1e-3
n_layer = 4
dropout = 0.0

# train/test split
train_portion = int(0.9 * len(text))
train_data = torch.tensor(encode(text[:train_portion]), dtype=torch.long)
test_data = torch.tensor(encode(text[train_portion:]), dtype=torch.long)

# data loading
def get_batch(split):
    data = train_data if split == 'train' else test_data
    idx = torch.randint(len(text) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    model.train()


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        head_embd = n_embd // n_head
        self.query = nn.Linear(head_size, head_embd)
        self.key = nn.Linear(head_size, head_embd)
        self.value = nn.Linear(head_size, head_embd)
        self.register_buffer('tril', torch.tril(torch.ones(head_embd, head_embd)).float())

    def forward(self, x):
        B,T,C = x.shape()

        q, k, v = self.query(x), self.key(x), self.value(x)

        q @ k.transpose(-2, -1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        heads = torch.ModuleList([Head(n_embd//head_size) for _ in range(num_heads)])
    def forward(self, x):
        return


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.mean = torch.zeros(features)
        self.std = torch.ones(features)
        
    def forward(self, x):
        return

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, idx, targets=None):
        loss = None
        if targets is not None:
            ;
        # return x
    
    def generate(self, idx, max_new_tokens):
        return
    

model = BigramLanguageModel()
m = model.to(device)
# print no. of parameters
print(sum([p.numel() for p in m.parameters()]) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)