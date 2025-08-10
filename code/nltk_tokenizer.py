import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required tokenizer models
nltk.download('punkt')

text = "I love NLP! It's amazing. Let's build a chatbot."

# 1. Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentence Tokenization:", sentences)

# 2. Word Tokenization
words = word_tokenize(text)
print("Word Tokenization:", words)
