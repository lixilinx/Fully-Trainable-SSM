import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from flash_attn import flash_attn_func
import time

# ---- Fast Convolution Implementation ----
def fast_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    len1, _ = a.shape
    _, len2, _ = b.shape
    T = 2**int(np.ceil(np.log2(len1 + len2 - 1)))
    A = torch.fft.fft(a, n=T, dim=0)
    B = torch.fft.fft(b, n=T, dim=1)
    C = A * B
    c = torch.fft.ifft(C, n=T, dim=1)
    return c[..., :(len1 + len2 - 1), :]


# ====================== Enhanced Components ======================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# ====================== Enhanced SSM Modules ======================
class CustomComplexSSM(nn.Module):
    def __init__(self, input_size: int, state_size: int, output_size: int):
        super().__init__()
        self.A = nn.Parameter(0.999 * torch.exp(torch.randn(state_size) * 1j))
        self.B = nn.Parameter(torch.randn(input_size, state_size, dtype=torch.complex64)/np.sqrt(state_size))
        self.C = nn.Parameter(torch.randn(state_size, output_size, dtype=torch.complex64)/np.sqrt(state_size))
        self.D = nn.Parameter(torch.randn(output_size))
        self.norm = RMSNorm(input_size)
        
    def forward(self, u: torch.Tensor):
        u = self.norm(u)
        batch_size, seq_len, _ = u.shape
        A_powers = torch.pow(self.A, torch.arange(seq_len, device=u.device)[:, None])
        uB = torch.einsum('bli,iz->blz', u.to(torch.complex64), self.B)
        x = fast_conv(A_powers, uB)[:, :seq_len]
        y = torch.real(torch.einsum('blz,zo->blo', x, self.C)) + self.D
        return y, x[:, -1]

class MambaSSM(nn.Module):
    def __init__(self, input_size: int, state_size: int, output_size: int):
        super().__init__()
        self.dt = nn.Parameter(torch.randn(state_size))
        self.A = nn.Parameter(torch.randn(state_size))
        self.B = nn.Parameter(torch.randn(input_size, state_size)/np.sqrt(state_size))
        self.C = nn.Parameter(torch.randn(state_size, output_size)/np.sqrt(state_size))
        self.D = nn.Parameter(torch.randn(output_size))
        self.norm = RMSNorm(input_size)
        
    def forward(self, u: torch.Tensor):
        u = self.norm(u)
        batch_size, seq_len, _ = u.shape
        dt = torch.sigmoid(self.dt)
        A_positive = torch.abs(self.A)
        A_disc = torch.exp(-dt * A_positive)
        t = torch.arange(seq_len, device=u.device).float().unsqueeze(0)
        h = dt.unsqueeze(1) * (A_disc.unsqueeze(1) ** t)
        u_proj = torch.einsum('bti,is->bts', u, self.B).transpose(1, 2)
        kernel = torch.flip(h, [1]).unsqueeze(1)
        x = F.conv1d(u_proj, kernel, padding=seq_len-1, groups=u_proj.shape[1])[:, :, :seq_len]
        x = x.transpose(1, 2)
        y = torch.einsum('bts,so->bto', x, self.C) + self.D
        return y, x[:, -1]



# ====================== Enhanced Language Models ======================
class TinySSMLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, state_size: int, ssm_type: str = "complex"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_norm = RMSNorm(embed_dim)
        
        if ssm_type == "complex":
            self.ssm = CustomComplexSSM(embed_dim, state_size, embed_dim)
        elif ssm_type == "mamba":
            self.ssm = MambaSSM(embed_dim, state_size, embed_dim)
            
        self.head = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embed_norm(self.embedding(input_ids))
        x, _ = self.ssm(x)
        return self.head(x)


# ====================== Training Setup ======================
class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer, seq_len: int):
        self.token_ids = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.token_ids[idx:idx+self.seq_len]),
            torch.tensor(self.token_ids[idx+1:idx+self.seq_len+1])
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention without masking
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final linear layer
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        # Multi-head attention
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # Feed-forward network
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class SmallAttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(SmallAttentionLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 128, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Embedding and positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output layer
        x = self.fc_out(x)
        return x

def train_model(optimizer, model, train_loader, device, model_name, num_steps=10_000, eval_every=100):
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    step = 0
    train_iter = iter(train_loader)
    
    while step < num_steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        def closure():
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        if (step+1) % eval_every == 0:
            avg_loss = np.mean(losses[-eval_every:])
            print(f"{model_name} - Step {step+1}: Loss = {avg_loss:.4f}")
        
        step += 1
    
    return losses

def get_attention_model_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Configuration
    SEQ_LEN = 128
    BATCH_SIZE = 64
    EMBED_DIM = 128
    STATE_SIZE = 256
    NUM_HEADS = 8
    NUM_LAYERS = 6  # Increased depth
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    with open("input.txt", "r", encoding="utf-8") as f: # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        text = f.read()

    # Prepare data
    train_ds = TextDataset(text, tokenizer, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    complex_model = TinySSMLM(tokenizer.vocab_size, EMBED_DIM, STATE_SIZE, "complex").to(device)
    mamba_model = TinySSMLM(tokenizer.vocab_size, EMBED_DIM, STATE_SIZE, "mamba").to(device)
    attention_model = SmallAttentionLanguageModel(tokenizer.vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(device)
    print('Complex Model Params: ', get_attention_model_param_count(complex_model))
    print('Mamba Model Params: ', get_attention_model_param_count(mamba_model))
    print('Attention Model Params: ', get_attention_model_param_count(attention_model))
    # Train all models


    # Train Complex SSM Model with timing
    print("Training Complex SSM Model...")
    torch.cuda.synchronize()  # Ensure previous GPU operations are complete
    start_time = time.time()
    optimizer = Kron(
        complex_model.parameters(), 
        lr_params=5e-3, 
        momentum=0.9, 
        lr_preconditioner=0.1, 
        preconditioner_update_probability=0.1, 
        preconditioner_type="whitening"
    )
    complex_losses = train_model(optimizer, complex_model, train_loader, device, "Complex SSM")
    torch.cuda.synchronize()  # Ensure all GPU computations have finished
    end_time = time.time()
    print(f"Complex SSM training completed in {end_time - start_time:.2f} seconds.")

    # Train Mamba SSM Model with timing
    print("\nTraining Mamba SSM Model...")
    torch.cuda.synchronize()
    start_time = time.time()
    optimizer = Kron(
        mamba_model.parameters(), 
        lr_params=5e-3, 
        momentum=0.9, 
        lr_preconditioner=0.1, 
        preconditioner_update_probability=0.1, 
        preconditioner_type="whitening"
    )
    mamba_losses = train_model(optimizer, mamba_model, train_loader, device, "Mamba SSM")
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Mamba SSM training completed in {end_time - start_time:.2f} seconds.")

    # Train FlashAttention Model with timing
    print("\nTraining FlashAttention Model...")
    torch.cuda.synchronize()
    start_time = time.time()
    optimizer = Kron(
        attention_model.parameters(), 
        lr_params=1e-3, 
        momentum=0.9, 
        lr_preconditioner=0.1, 
        preconditioner_update_probability=0.1, 
        preconditioner_type="whitening"
    )
    attn_losses = train_model(optimizer, attention_model, train_loader, device, "FlashAttention")
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"FlashAttention training completed in {end_time - start_time:.2f} seconds.")

    # Plot results
    plt.figure(figsize=(12, 6))
    # Convert losses to perplexity (PPL)
    complex_ssm_ppl = np.exp(complex_losses)
    mamba_ssm_ppl = np.exp(mamba_losses)
    attn_ppl = np.exp(attn_losses)

    plt.plot(complex_ssm_ppl, label="Complex SSM", linestyle='-', marker='')
    plt.plot(mamba_ssm_ppl, label="Mamba SSM", linestyle='-', marker='')
    plt.plot(attn_ppl, label='FlashAttention', linestyle='-', marker='')

    plt.ylim(0, 5)
    plt.xlabel("Steps")
    plt.ylabel("Perplexity (PPL)")
    plt.title("Training Loss Curve (Perplexity)")
    plt.legend()
    plt.grid(True)
    plt.show()    

if __name__ == "__main__":
    main()
