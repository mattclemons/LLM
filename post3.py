import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# Assuming the Transformer model from Post 2
class Transformer(nn.Module):
    def __init__(self, embed_size, heads, depth, forward_expansion, max_len, dropout, vocab_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = SimplePositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, forward_expansion, dropout) for _ in range(depth)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, x, x)

        out = self.fc_out(x)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention_weights = torch.softmax(attention / (self.embed_size ** 0.5), dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention_weights, values]).reshape(
            N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SimplePositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(SimplePositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)

        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                self.encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / embed_size)))
                self.encoding[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / embed_size)))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :]

# Load and tokenize dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentences = ["The quick brown fox jumps over the lazy dog.", "The Transformers architecture is very powerful."]
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Target sequence for training (random example for simplicity)
target_ids = torch.tensor([[101, 2023, 2003, 1037, 3944, 102], [101, 2057, 2024, 1037, 2037, 102]])

# Define the optimizer and loss function
optimizer = optim.Adam(Transformer.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
def train(model, input_ids, target_ids, attention_mask, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_ids)
        outputs = outputs.view(-1, outputs.size(-1))
        target_ids = target_ids.view(-1)
        loss = loss_fn(outputs, target_ids)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Example of training
model = Transformer(embed_size=256, heads=8, depth=4, forward_expansion=4, max_len=50, dropout=0.1, vocab_size=30522)
train(model, input_ids, target_ids, attention_mask)

# Monitoring loss during training with plot
def train_with_monitoring(model, input_ids, target_ids, attention_mask, epochs=10):
    model.train()
    loss_values = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_ids)
        outputs = outputs.view(-1, outputs.size(-1))
        target_ids = target_ids.view(-1)
        loss = loss_fn(outputs, target_ids)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    plt.plot(range(1, epochs + 1), loss_values, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Run training with monitoring
train_with_monitoring(model, input_ids, target_ids, attention_mask)

# Evaluation function
def evaluate(model, input_ids, target_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        outputs = outputs.view(-1, outputs.size(-1))
        target_ids = target_ids.view(-1)
        loss = loss_fn(outputs, target_ids)
        print(f"Evaluation Loss: {loss.item()}")

# Example of evaluation
evaluate(model, input_ids, target_ids, attention_mask)

