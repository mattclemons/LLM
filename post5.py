import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from fastapi import FastAPI
import uvicorn
import math

# Transformer model definition
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

# Positional Encoding
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

# Transformer block definition
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

# Multi-head self-attention
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

# Feedforward layer
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Save and load model functions
def save_model(model, path="transformer_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="transformer_model.pth"):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  # Ensure the path is correct
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")

# Create and initialize the FastAPI app
app = FastAPI()

# Load the model and tokenizer
model = Transformer(embed_size=256, heads=8, depth=4, forward_expansion=4, max_len=50, dropout=0.1, vocab_size=30522)

# Save the model first (if not already saved from previous training)
save_model(model, path="transformer_model.pth")

# Load the model (this will work now since the model has been saved)
load_model(model, path="transformer_model.pth")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.post("/predict/")
async def predict_text(input_text: str):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_ids)
        predicted_token_id = torch.argmax(outputs, dim=-1).item()

    # Convert the predicted token ID back to a word
    predicted_word = tokenizer.decode(predicted_token_id)

    return {"input": input_text, "prediction": predicted_word}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

