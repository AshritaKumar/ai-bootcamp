from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
texts = ["I love NLP", "I hate bugs", "AI is amazing", "I dislike errors"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train a simple model
model = LogisticRegression()
model.fit(X, labels)

# Prediction
print(model.predict(vectorizer.transform(["I love AI"])))
