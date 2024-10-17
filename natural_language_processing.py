# Import NLTK and download its tokenizer data:
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Now, tokenize a sentence:
from nltk.tokenize import word_tokenize

text = "Hello! How are you today?"
tokens = word_tokenize(text)
print(tokens)

# Example: Removing Stopwords
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)

# Example: Tokenizing with BERT
from transformers import AutoTokenizer

# Load the pre-trained BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize a sentence
text = "Hello! How are you today?"
tokens = tokenizer.tokenize(text)
print(tokens)

# Convert tokens to token IDs (which the model uses for input):
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

