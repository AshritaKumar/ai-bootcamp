import re
from transformers import AutoTokenizer

text = "I love NLP, a lot!"

# Whitespace
print("Whitespace:", text.split())

# Regex-based
print("Regex:", re.findall(r'\w+', text))

# WordPiece (BERT)
tokenizer_wp = AutoTokenizer.from_pretrained("bert-base-uncased")
print("WordPiece:", tokenizer_wp.tokenize(text))

# Byte-Level BPE (GPT-2)
tokenizer_bpe = AutoTokenizer.from_pretrained("gpt2")
print("Byte-Level BPE:", tokenizer_bpe.tokenize(text))
