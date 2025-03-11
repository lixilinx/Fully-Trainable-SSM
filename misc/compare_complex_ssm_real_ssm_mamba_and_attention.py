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

def fast_conv_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Inputs:
        a: tensor with shape [len1, K, n, n] representing len1 block diagonal matrices, 
           each block has size n x n, and a total of K such blocks. 
        b: another torch tensor with shape [batch, len2, K, n] representing K vectors, 
           and each has length n.
        
    Outputs:
        c = conv(b, a) with shape [batch, len1 + len2 - 1, K, n]. 
        Following the 'row major' convention, here it's the vector-matrix product. 
        So I put a after b. 
    """
    len1, _, _, _ = a.shape
    _, len2, _, _ = b.shape
    T = 2**int(np.ceil(np.log2(len1 + len2 - 1)))
    A = torch.fft.rfft(a, n=T, dim=0)
    B = torch.fft.rfft(b, n=T, dim=1)
    C = torch.einsum("blki, lkij->blkj", B, A)
    c = torch.fft.irfft(C, n=T, dim=1)
    return c[:, :(len1 + len2 - 1)]


def A_powers(A: torch.Tensor, t: int) -> torch.Tensor:
    """
    Inputs:
        A: tensor with shape [K, n, n] representing a block diagonal matrices, 
           each block has size n x n, and a total of K such blocks.
        t: a positive integer.
        
    Outputs:
        torch.stack([I, A, A^2, ..., A^(t - 1)], dim=0), 
        a tensor with shape [t, K, n, n]. 
    """
    K, n, _ = A.shape 
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    result = torch.stack([eye.repeat(K, 1, 1), A])
    lift = A @ A
    for _ in range(int(np.ceil(np.log2(t))) - 1):
        result = torch.cat([result, result @ lift])
        lift = lift @ lift
    return result[:t]


class RealStateSpaceModel(torch.nn.Module):
    """
    It returns a fully trainable real state SSM defined as:
        
        x_t = A @ x_{t-1} + B @ u_t,
        y_t = C @ x_t + D @ u_t + b,
        
    where:
        
        u_t, 1 <= t <= T, are the sequences of (real) inputs,
        x_t, 1 <= t <= T, are the sequence of (real) states,
        y_t, 1 <= t <= T, are the sequence of (real) outputs,
        matrices A (block diagonal real), B (real) and C (real) are mandatory, 
        matrix D (real) and bias b (real) are optional,
        and x_0 is the initial (real) state. 
        
    Note that we use the tranposed version of these equations in the Python code 
    by following the 'row major' convention.  
    """ 
    def __init__(self, input_size: int, state_size: int, output_size: int,  
                 has_matrixD: bool=False, has_bias: bool=False, decimation: int=1,
                 init_scale_A: float=1 - 1e-4) -> None:
        """
        Inputs:
            input_size, state_size and output_size: sizes of u, x, and y, respectively, 
                    and the state_size must be an even number.
            has_matrixD: matrix D is None if setting to False, otherwise not. 
            has_bias: bias b is None if setting to False, otherwise not. 
            decimation: decimate states and outputs if > 1 (same as the stride parameter in CNN),
                        interpolate states if < 1 (ToDo; similar to tranposed CNN).
            init_scale_A: set to < 1 such that the SSM is stable. 
        """
        super(RealStateSpaceModel, self).__init__()
        state_blk_size = 2 # only value 2 makes sense in theory. 
        assert(state_size%state_blk_size == 0)
        self.state_blk_size = state_blk_size
        self.state_num_blks = state_size // state_blk_size
        theta = 2*torch.pi*torch.rand(state_size//2)
        A = torch.zeros(self.state_num_blks, 2, 2)
        A[:,0,0] = torch.cos(theta)
        A[:,0,1] = torch.sin(theta)
        A[:,1,0] = -torch.sin(theta)
        A[:,1,1] = torch.cos(theta)
        self.A = torch.nn.Parameter(init_scale_A * A) 
        self.B = torch.nn.Parameter(
            torch.randn(input_size, state_size)/(state_size + input_size)**0.5)
        self.C = torch.nn.Parameter(
            torch.randn(state_size, output_size)/state_size**0.5)
        self.has_matrixD = has_matrixD
        if has_matrixD:
            self.D = torch.nn.Parameter(torch.zeros(input_size, output_size))
        self.has_bias = has_bias
        if has_bias:
            self.b = torch.nn.Parameter(torch.zeros(output_size))
        self.decimation = decimation
        self.norm = RMSNorm(input_size)


    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):
        """
        Inputs:
            u: the real input tensor with shape [batch, length, input_size].
            x0: the real initial state with shape [batch, state_num_blk, state_blk_size]. 
            
        Outputs:
            y: the real output tensor with shape [batch, length, output_size].
            x0: the real final state with shape [batch, state_num_blk, state_blk_size]. 
        """            
        u = self.norm(u)
        _, length, _ = u.shape
        Aps = A_powers(self.A, length + 1)
        uB = u @ self.B # tranpose of math eq B*u
        uB = torch.reshape(uB, (-1, length, self.state_num_blks, self.state_blk_size))
        x = fast_conv_real(Aps[:-1], uB)[:, :length]
        if x0 is not None:
            x = x + torch.einsum("bki,lkij->blkj", x0, Aps[1:])
        new_state = x[:, -1]
        x = torch.reshape(x, (-1, length, self.state_num_blks * self.state_blk_size))
        if self.decimation > 1: # decimate to length length//decimation
            x = x[:, self.decimation-1::self.decimation]
            if self.has_matrixD:
                u = u[:, self.decimation-1::self.decimation]
        y = x @ self.C # tranpose of math eq C * x
        if self.has_matrixD:
            y = y + u @ self.D # transpose of math eq D * u 
        if self.has_bias:
            y = y + self.b
        return (y, new_state)



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
        elif ssm_type == "real":
            self.ssm = RealStateSpaceModel(embed_dim, state_size, embed_dim)
            
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
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Prepare data
    train_ds = TextDataset(text, tokenizer, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    complex_model = TinySSMLM(tokenizer.vocab_size, EMBED_DIM, STATE_SIZE, "complex").to(device)
    real_model = TinySSMLM(tokenizer.vocab_size, EMBED_DIM, STATE_SIZE, "real").to(device)
    mamba_model = TinySSMLM(tokenizer.vocab_size, EMBED_DIM, STATE_SIZE, "mamba").to(device)
    attention_model = SmallAttentionLanguageModel(tokenizer.vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(device)
    print('Complex Model Params: ', get_attention_model_param_count(complex_model))
    print('Real Model Params: ', get_attention_model_param_count(real_model))
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


    # Train Real SSM Model with timing
    print("Training Real SSM Model...")
    torch.cuda.synchronize()  # Ensure previous GPU operations are complete
    start_time = time.time()
    optimizer = Kron(
        real_model.parameters(), 
        lr_params=5e-3, 
        momentum=0.9, 
        lr_preconditioner=0.1, 
        preconditioner_update_probability=0.1, 
        preconditioner_type="whitening"
    )
    real_losses = train_model(optimizer, real_model, train_loader, device, "Real SSM")
    torch.cuda.synchronize()  # Ensure all GPU computations have finished
    end_time = time.time()
    print(f"Real SSM training completed in {end_time - start_time:.2f} seconds.")

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
    real_ssm_ppl = np.exp(real_losses)
    mamba_ssm_ppl = np.exp(mamba_losses)
    attn_ppl = np.exp(attn_losses)

    plt.plot(complex_ssm_ppl, label="Complex SSM", linestyle='-', marker='')
    plt.plot(real_ssm_ppl, label="Real SSM", linestyle='-', marker='')
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
