from transformers import AutoTokenizer, AutoModel, pipeline

# Load pipeline for feature extraction
model_name = "bert-base-uncased"
feature_extractor = pipeline("feature-extraction", model=model_name)

text = "I love NLP"

# Get embeddings
embeddings = feature_extractor(text)

# embeddings is a 3D list: [batch, tokens, hidden_size]
# Let's average over tokens for a single vector
sentence_embedding = [sum(token) / len(token) for token in zip(*embeddings[0])]

print("Embedding length:", len(sentence_embedding))
# print("First 5 values:", sentence_embedding[:5])
print("Full embedding:", sentence_embedding)