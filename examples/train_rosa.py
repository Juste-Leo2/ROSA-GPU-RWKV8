# File: train_rosa_cpp_backend.py
# Description: Script to train the GPU-optimized ROSA model using the compiled C++ backend
# for maximum training speed.

import torch
import random
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import rosa_cuda_ext # Import the compiled C++ extension

# --- Configuration ---
# Recommended performance settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model and training hyperparameters
V, C, B, T = 11, 64, 128, 128
TRAINING_STEPS = 1000
LEARNING_RATE = 3e-3

print(f"Using device: {device}")
print("Starting training with C++ CUDA backend to reproduce the loss curve...")

####################################################################################################
# --- Model and Data Definitions (Adapted for C++ Backend) ---
####################################################################################################

class rosa_emb_layer(nn.Module):
    # This layer uses the ROSA algorithm on the input token indices
    # before feeding them to the embedding table.
    def __init__(self, V, C):
        super().__init__()
        self.emb = nn.Embedding(V, C)

    def forward(self, idx):
        # Use the ultra-fast C++ CUDA extension instead of the Python/PyTorch version.
        # This is the primary source of speed-up.
        idx_pred = rosa_cuda_ext.forward(idx)
        
        # Get embeddings for the predicted indices.
        out = self.emb(idx_pred.clamp_min(0))
        
        # Mask out positions where ROSA made no prediction (-1) by setting them to zero.
        return out.masked_fill(idx_pred.eq(-1).unsqueeze(-1), 0.0)

class rosa_4bit_layer(nn.Module):
    # This layer quantizes the input tensor into 4-bit representations,
    # applies the ROSA algorithm on these representations, and then
    # de-quantizes the result using learned embeddings.
    def __init__(self, C: int, eps: float = 1e-5):
        super().__init__()
        assert C % 4 == 0
        # Learnable embeddings for "0" and "1" bits.
        self.emb0 = nn.Parameter(torch.full((1, 1, C), -eps))
        self.emb1 = nn.Parameter(torch.full((1, 1, C),  eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape; Cg = C // 4
        
        # 1. Quantize: Convert input tensor to 4-bit integers.
        # Each group of 4 channels becomes a single 4-bit number (0-15).
        b = (x.reshape(B, T, Cg, 4) > 0).to(torch.uint8)
        tok2d = (b[...,0] | (b[...,1] << 1) | (b[...,2] << 2) | (b[...,3] << 3)).permute(0, 2, 1).reshape(-1, T).contiguous()

        # 2. Predict: Apply the ROSA algorithm using the fast C++ CUDA backend.
        # FIX: Convert tok2d to torch.long to match the C++ extension's requirement.
        idx_q = rosa_cuda_ext.forward(tok2d.to(torch.long)).reshape(B, Cg, T).transpose(1, 2).contiguous()

        # 3. De-quantize: Use the predicted 4-bit values to select between emb0 and emb1.
        e0 = self.emb0.expand(B, T, -1).reshape(B, T, Cg, 4)
        e1 = self.emb1.expand(B, T, -1).reshape(B, T, Cg, 4)
        bits = torch.stack([(idx_q >> i) & 1 for i in range(4)], dim=-1).bool()
        
        return torch.where(bits, e1, e0).reshape(B, T, C)


def batch(B, T):
    # Generates a batch of data. Each sequence is a series of numbers
    # separated by a special token (10).
    s = []
    for _ in range(B):
        k = random.randint(1, 3)
        lo = 0 if k == 1 else 10**(k - 1)
        n = random.randint(lo, 10**k - 1)
        a = [10]
        while len(a) < T + 1: # Generate one extra token for the target
            a += [ord(c) - 48 for c in str(n)] + [10]
            n += 1
        s.append(a[:T + 1])
    return torch.tensor(s, device=device, dtype=torch.long)

class MODEL(nn.Module):
    # The main model architecture, identical to the original implementation.
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(V,C)
        self.emb_rosa = rosa_emb_layer(V,C)
        self.rosa1 = rosa_4bit_layer(C)
        self.lin = nn.Linear(C,C)
        self.rosa2 = rosa_4bit_layer(C)
        self.lin1 = nn.Linear(C,C)
        self.rosa3 = rosa_4bit_layer(C)
        self.lin2 = nn.Linear(C,C)
        self.rosa4 = rosa_4bit_layer(C)
        self.o = nn.Linear(C,V)
        
    def forward(self, x):
        # The forward pass remains unchanged.
        x = self.e(x) + self.emb_rosa(x)
        x = x + self.rosa1(x)
        x = x + self.lin(x)
        x = x + self.rosa2(x)
        x = x + self.lin1(x)
        x = x + self.rosa3(x)
        x = x + self.lin2(x)
        x = x + self.rosa4(x)
        x = self.o(x)
        return x

####################################################################################################
# --- Training Loop ---
####################################################################################################

# Initialize model, optimizer, and loss function
model = MODEL().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# List to store the loss at each step
loss_history = []

model.train() # Set the model to training mode
progress_bar = tqdm(range(TRAINING_STEPS), desc="Training Progress")

for step in progress_bar:
    # 1. Prepare data
    data_batch = batch(B, T)
    inputs = data_batch[:, :-1].contiguous()
    targets = data_batch[:, 1:].contiguous()

    # 2. Zero the gradients
    optimizer.zero_grad()

    # 3. Forward pass
    logits = model(inputs)

    # 4. Calculate loss
    # Reshape logits and targets for CrossEntropyLoss
    loss = criterion(logits.view(-1, V), targets.view(-1))

    # 5. Backward pass and optimization
    loss.backward()
    optimizer.step()

    # 6. Store and display the loss
    loss_value = loss.item()
    loss_history.append(loss_value)
    progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

print("Training finished.")

####################################################################################################
# --- Plot Generation ---
####################################################################################################

print("Generating the loss plot...")

plt.figure(figsize=(12, 7))
plt.plot(loss_history, label='EmbROSA + 4bit x 4layer (C++ Backend)', color='#FFC300', linewidth=1.5)

# Customize to resemble the original plot
plt.title("ROSA Model Learning Curve", fontsize=16)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.ylim(0, 1.0)
plt.xlim(0, TRAINING_STEPS)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)

# Save the plot to a file
output_filename = 'training_loss_plot_cpp.png'
plt.savefig(output_filename)

print(f"Plot saved to: {output_filename}")
# plt.show() # Uncomment this line to display the plot directly