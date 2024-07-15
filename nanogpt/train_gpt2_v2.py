import math
import time
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

start = time.time()
def timer_end():
    print(f"time taken using {device}: {(time.time() - start):.4f}", )

@dataclass
class GPTConfig:
    block_size: int = 1024      # max sequence length
    vocab_size: int = 50257     # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12           # number of layers
    n_head: int = 12            # number of heads
    n_embd: int = 768           # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # notice the view(1,1,..) to make it broadcastable

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        # q,k,v = qkv.split(self.n_embd, dim=2) # NOTE: splitting last dimension into n_embd chunks, TODO: check dimensions
        q,k,v = qkv.chunk(3, dim=2) # NOTE: alternate way of splitting the last dimension into n_embd chunks
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # B, T, C -> B, T, n_head, C//n_head -> B, n_head, T, C//n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) * (1 / math.sqrt(k.size(-1))) # TODO: see if q @ k.transpose(-2,-1) gives the same result as q.transpose(1,2) @ k, see if k.size(-1) is the same as n_embd
        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        logits = F.softmax(attn, dim=-1)
        out = (logits @ v).transpose(1,2).contiguous().view(B, T, C) # TODO: see what the contiguous() function does
        return self.c_proj(out)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        # NOTE: ModuleDict is a dictionary that holds submodules
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # the x returned is one softmax away from becoming probabilities
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    
    # COPIED
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a HuggingFace/Transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# --------------------------------------------------------------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens\n{len(self.tokens)//(B*T)} batches of size {B}x{T}")
        
        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1] # +1 because of the target token
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y
# --------------------------------------------------------------------------------------------------------------------------------

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("device:", device)

train_loader = DataLoaderLite(B=4, T=32)

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
# print("didn't crash yay")

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(10):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i+1}, loss: {loss.item():.4f}")

timer_end()

import sys; sys.exit(0)
model.eval()
num_return_sequences = 5
max_length = 30
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) # repeats at dimension 1 for num_return_sequences times
x = tokens.to(device)

torch.manual_seed(42)
torch.mps.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        out = model(x) # (B, T, vocab_size)
        # take the logits only at the last position (works but wasteful)
        out = out[:,-1,:] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(out, dim=-1) # performs softmax in the dimension with vocab_size
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        selected_index = torch.multinomial(topk_probs, num_samples=1)
         # (8, 1)
        # gather the corresponding indices
        selected_token = topk_indices.gather(dim=1, index=selected_index) # NOTE: look into the gather function, it's like fancy indexing
         # (8, 1)
        # append to the sequence
        x = torch.cat((x, selected_token), dim=1)
         # (8, T+1)

# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)