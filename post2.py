import torch
import torch.nn as nn
import math

# Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # Split the embedding size across the heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        # Linear layers to project the inputs into queries, keys, and values
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embeddings into multiple heads for parallel processing
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Calculate energy (dot product attention)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Normalize the attention scores
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

        # Get weighted sum of values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Apply the final linear layer
        out = self.fc_out(out)
        return out

# Feedforward Neural Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)

        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                self.encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / embed_size)))
                self.encoding[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / embed_size)))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].to(x.device)

# Transformer Block with Layer Normalization and Residual Connections
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        # Multi-head attention
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        # Feedforward network
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        # Apply multi-head attention and add residual connection
        attention = self.attention(value, key, query)
        x = self.norm1(attention + query)
        
        # Apply feedforward and another residual connection
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

# Full Transformer Model
class Transformer(nn.Module):
    def __init__(self, embed_size, heads, depth, forward_expansion, max_len, dropout, vocab_size):
        super(Transformer, self).__init__()
        # Embedding layer to convert words to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Positional encoding to remember word order
        self.pos_encoding = PositionalEncoding(embed_size, max_len)
        # Stack multiple Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, forward_expansion, dropout) for _ in range(depth)]
        )
        # Final output layer to predict the next word
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # Convert input words to embeddings
        x = self.embedding(x)
        # Add positional encodings
        x = self.pos_encoding(x)

        # Pass through each Transformer block
        for layer in self.layers:
            x = layer(x, x, x)

        # Predict the output
        out = self.fc_out(x)
        return out

# Example usage of the Transformer model
def run_example():
    # Parameters
    vocab_size = 100  # Number of unique words
    embed_size = 8    # Size of word embeddings
    heads = 2         # Number of attention heads
    depth = 3         # Number of layers in the Transformer
    forward_expansion = 4  # Expansion factor in the feedforward network
    max_len = 10      # Maximum sentence length
    dropout = 0.1     # Dropout rate to prevent overfitting

    # Create a Transformer model
    model = Transformer(embed_size, heads, depth, forward_expansion, max_len, dropout, vocab_size)

    # Example input: a batch of 1 sentence with 10 words (random word indices)
    input_sentence = torch.randint(0, vocab_size, (1, max_len))  # Random input sentence

    # Get the model's output
    output = model(input_sentence)

    # Print input and output
    print("Input sentence (word indices):")
    print(input_sentence)
    print("\nModel output (raw scores for each word in the vocabulary):")
    print(output)

if __name__ == "__main__":
    run_example()

