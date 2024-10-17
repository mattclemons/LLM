import torch
import torch.nn as nn
import math

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
    def __init__(self, embed_size, heads, depth, forward_expansion, max_len, dropout):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(depth)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(self.positional_encoding(x))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

# Example of how to instantiate and run the model
if __name__ == "__main__":
    embed_size = 512
    heads = 8
    depth = 6
    forward_expansion = 4
    max_len = 100
    dropout = 0.1
    batch_size = 64
    seq_len = 100

    model = Transformer(embed_size, heads, depth, forward_expansion, max_len, dropout)

    # Create random input (batch of sequences)
    inputs = torch.rand(batch_size, seq_len, embed_size)
    mask = torch.ones(batch_size, seq_len)

    # Forward pass
    outputs = model(inputs, mask)
    print(outputs.shape)  # Should output [batch_size, seq_len, embed_size]

