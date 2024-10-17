from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load a pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Tokenize input text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Pass the tokenized inputs to the model
with torch.no_grad():  # Disable gradient computation for inference
    outputs = model(**inputs)
    logits = outputs.logits
    print(logits)

# Experiment with text generation using GPT-2, a model designed to generate sequences
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Tokenize a text prompt
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
with torch.no_grad():
    output = model.generate(inputs['input_ids'], max_length=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

# Get Embeddings from BERT
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize input
text = "Transformers are great!"
inputs = tokenizer(text, return_tensors="pt")

# Get the hidden states (embeddings)
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state

print(hidden_states)

