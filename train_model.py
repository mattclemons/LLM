import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Calculate energy (dot product attention mechanism)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Expand the mask to match the attention energy shape
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # Weighted sum of values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# Feedforward Neural Network Layer
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, embed_size, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        return x + self.encoding[:seq_len, :].to(x.device)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Full Transformer Model
class Transformer(nn.Module):
    def __init__(self, embed_size, heads, depth, forward_expansion, max_len, dropout, vocab_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(depth)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer maps to vocab size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Convert input token indices into embeddings
        x = self.embedding(x)
        out = self.dropout(self.positional_encoding(x))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc_out(out)  # Predict vocab tokens
        return out

# Dataset Class
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly generate an input sequence
        input_seq = torch.randint(0, self.vocab_size, (self.seq_len,))
        # The target is the input sequence shifted by one
        target_seq = torch.cat((input_seq[1:], torch.tensor([0])))  # Shifted next token prediction
        return input_seq, target_seq

# Hyperparameters and Dataset Initialization
num_samples = 1000
seq_len = 10
vocab_size = 50
embed_size = 512
heads = 8
depth = 6
forward_expansion = 4
max_len = 100
dropout = 0.1
epochs = 5

dataset = SyntheticDataset(num_samples, seq_len, vocab_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Optimizer, and Loss Function
model = Transformer(embed_size, heads, depth, forward_expansion, max_len, dropout, vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(torch.long), targets.to(torch.long)

        # Forward pass
        outputs = model(inputs, None)  # No mask needed in this case
        outputs = outputs.view(-1, vocab_size)  # Reshape for loss calculation
        targets = targets.view(-1)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

# Model Evaluation (Sequence Generation)
model.eval()
with torch.no_grad():
    input_seq = torch.randint(0, vocab_size, (1, seq_len)).long()  # A random input sequence
    output_seq = model(input_seq, None)
    predicted_tokens = torch.argmax(output_seq, dim=-1)
    print("Input sequence:", input_seq)
    print("Predicted next tokens:", predicted_tokens)

