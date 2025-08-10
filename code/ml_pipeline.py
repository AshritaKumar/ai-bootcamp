from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
texts = ["I love NLP", "I hate bugs", "AI is amazing", "I dislike errors"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative
label_names = {1: "Positive", 0: "Negative"}  # Map numeric labels to text

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train a simple model
model = LogisticRegression()
model.fit(X, labels)

# Test sentence
test_sentence = "I love AI"
prediction = model.predict(vectorizer.transform([test_sentence]))[0]

# Output
print(f"Input: {test_sentence}")
print(f"Predicted Sentiment: {label_names[prediction]}")
