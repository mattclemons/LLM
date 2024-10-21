import torch
import torch.nn as nn
import math

# Self-Attention Layer
class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SimpleSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        attention = torch.matmul(query, keys.transpose(-2, -1))
        attention_weights = torch.softmax(attention / (self.embed_size ** 0.5), dim=-1)
        out = torch.matmul(attention_weights, values)
        return out

# Feedforward Layer
class SimpleFeedForward(nn.Module):
    def __init__(self, embed_size):
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size * 2)
        self.fc2 = nn.Linear(embed_size * 2, embed_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Positional Encoding
class SimplePositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(SimplePositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                self.encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                self.encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / embed_size)))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].to(x.device)

# Transformer Block
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_size):
        super(SimpleTransformerBlock, self).__init__()
        self.attention = SimpleSelfAttention(embed_size)
        self.feed_forward = SimpleFeedForward(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

# Full Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, vocab_size, max_len):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = SimplePositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList([SimpleTransformerBlock(embed_size) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, x, x)
        out = self.fc_out(x)
        return out

# Sample usage
def run_example():
    # Parameters
    vocab_size = 100  # Vocabulary size (number of unique words)
    embed_size = 8    # Size of word embeddings
    num_layers = 2    # Number of Transformer layers
    max_len = 10      # Maximum sentence length
    batch_size = 1    # Number of sentences

    # Create a Transformer model
    model = SimpleTransformer(embed_size, num_layers, vocab_size, max_len)

    # Sample input: a batch of 1 sentence, with each word represented by a number
    input_sentence = torch.randint(0, vocab_size, (batch_size, max_len))  # Random sentence

    # Pass the sentence through the Transformer model
    output = model(input_sentence)

    # Print input and output
    print("Input sentence (as word indices):")
    print(input_sentence)
    print("\nModel output (raw scores for each word in the vocabulary):")
    print(output)

if __name__ == "__main__":
    run_example()

